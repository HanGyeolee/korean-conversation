# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:18:10 2022

@author: Han
"""
from __future__ import absolute_import

import os.path
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from tokenizer import Tokenizer
from mecm import StructureDataset
from mecm import MECM

tokenizer = Tokenizer(dicpath=r'../vocab.txt', update=False)
scaler = GradScaler()

#랜덤 시드 고정
SEED = 20220826
random.seed(SEED)
torch.manual_seed(SEED)

MAX_SEN = 512
#현재 단어는 634409개 이것보다 크게 잡은 이유는 나중에 단어 추가될 것을 대비
VOCAB_SIZE = 655350
EMBEDDING_DIM = 43
HIDDEN_DIM = 8
N_LAYERS = 64
DROPOUT = 0.25

# csv 파일로 학습 데이터 가져오기
train_data = StructureDataset(maxsize=MAX_SEN, csv_file='elemen_train_dataset.csv', tokenizer=tokenizer)
length = len(train_data)

batch_size = 100

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
# 모델 초기화
model = MECM(vocab_size=VOCAB_SIZE,
             embedding_dim=EMBEDDING_DIM,
             hidden_dim=HIDDEN_DIM,
             n_layers=N_LAYERS,
             dropout=DROPOUT).cuda()

PATH = "mecm_model_checkpoint.pt"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.AdamW(model.parameters())
criterion = nn.NLLLoss(ignore_index = 0)

e_epoch = 0;
# 기존 모델 불러오기
if os.path.exists(PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    e_epoch = checkpoint['epoch'];
    
    del checkpoint

def train(epoch):
    model.train()
    model.zero_grad()
    
    for batch_idx, (sentences, tags, lengths) in enumerate(train_loader):
        ll = 0
        count = (batch_idx + 1) * batch_size
        for idx in range(0, sentences.size(dim = 0)):
            sentence = sentences[idx,:lengths[idx]]
            tag = tags[idx,:lengths[idx]]
            
            with autocast():
                prediction = model(sentence) # sentences.squeeze()
                loss = criterion(prediction, tag) # tags.squeeze()
                
            #loss.backward()
            scaler.scale(loss / batch_size).backward()
            
            ll = loss.data.cpu().numpy()
                
            del sentence
            del tag
            del prediction
            del loss
            torch.cuda.empty_cache()
           
        if count >= length:
            count = length
        print('Train Epoch: {:<4} [{:>6}/{:<6} ({:3.0f}%)] Loss: {:.6f} GPU: {:5.2f}MiB'.format(
            epoch, 
            count ,
            length,
            100. * count / length, 
            ll,
            torch.cuda.memory_allocated()/1024/1024))
            
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        model.zero_grad()
            
        torch.cuda.empty_cache()

if True or not (os.path.exists(PATH)):
    #try:
    for epoch in range(2000):
        e_epoch += 1
        train(e_epoch)
    #except:
    #    pass
    
    torch.save({
        'epoch':e_epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
                }, PATH)

# 결과 테스트 
string = input("입력:")
now = datetime.datetime.now()
print(now.strftime('%Y.%m.%d.%w.%H:%M:%S.%f'))

tokenizer.tokenizing(string, allattrs=False)
tokens = tokenizer.tokens

prediction = model(torch.tensor(tokens).cuda())

def getstructure(prediction):
    dict_element = {0:'PAD',1:'EOF',2:'V',3:'S',4:'T',5:'Wy',6:'H',7:'WS',8:'WE',9:'DO',10:'IO',11:'Wi',12:'CO',13:'Adv'}
    structure = prediction.tolist()
    for i, sub in enumerate(structure):
        structure[i] = sub.index(max(sub))
        
    return [dict_element[v] for v in structure]
    
print(getstructure(prediction))

if True:
    del prediction
    del optimizer
    del scaler
    del tokenizer
    del train_data
    del train_loader
    torch.cuda.empty_cache()
