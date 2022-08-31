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
from mecm_reverse import StructureDataset
from mecm_reverse import MECM

tokenizer = Tokenizer(dicpath=r'../vocab.txt', update=False)
scaler = GradScaler()

# 랜덤 시드 고정
SEED = 20220826
random.seed(SEED)
torch.manual_seed(SEED)

MAX_SEN = 512
# 현재 단어는 634409개 이것보다 크게 잡은 이유는 나중에 단어 추가될 것을 대비
VOCAB_SIZE = 655350
EMBEDDING_DIM = 43
N_LAYERS = 64
DROPOUT = 0.2

# csv 파일로 학습 데이터 가져오기
train_data = StructureDataset(maxsize=MAX_SEN, csv_file='elemen_train_dataset.csv', tokenizer=tokenizer)
length = len(train_data)

batch_size = 100

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 모델 초기화
model = MECM(vocab_size=VOCAB_SIZE,
             embedding_dim=EMBEDDING_DIM,
             n_layers=N_LAYERS,
             dropout=DROPOUT).cuda()

PATH = "mecm_model_checkpoint.pt"


def count_parameters(p_model):
    return sum(p.numel() for p in p_model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.AdamW(model.parameters())
criterion = nn.NLLLoss(ignore_index=0).cuda()

e_epoch = 0;
# 기존 모델 불러오기
if os.path.exists(PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    e_epoch = checkpoint['epoch'];

    del checkpoint


def train(e):
    model.train()
    model.zero_grad()

    for batch_idx, (sentences, tags, lengths) in enumerate(train_loader):
        count = (batch_idx + 1) * batch_size
        with autocast():
            predictions = model(sentences)  # sentences.squeeze()
            loss = criterion(predictions, tags)  # tags.squeeze()

        # loss.backward()
        scaler.scale(loss).backward()
        ll = loss.data.cpu().numpy()

        del predictions
        del loss

        if count >= length:
            count = length

        print('Train Epoch: {:<4} [{:>6}/{:<6} ({:3.0f}%)] Loss: {:.6f} GPU: {:5.2f}MiB'.format(
            e,
            count,
            length,
            100. * count / length,
            ll,
            torch.cuda.memory_allocated() / 1024 / 1024))

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        model.zero_grad()

        torch.cuda.empty_cache()


if True or not (os.path.exists(PATH)):
    # try:
    for epoch in range(10):
        now = datetime.datetime.now()
        print(now.strftime('%Y.%m.%d.%w.%H:%M:%S.%f'))
        e_epoch += 1
        train(e_epoch)
    # except:
    #    pass

    torch.save({
        'epoch': e_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, PATH)

# 결과 테스트 
string = input("입력:")
now = datetime.datetime.now()
print(now.strftime('%Y.%m.%d.%w.%H:%M:%S.%f'))

tokenizer.tokenizing(string, allattrs=False)
tokens = tokenizer.tokens

prediction = model(torch.tensor(tokens).cuda())


def getstructure(pred):
    dict_element = {0: 'PAD', 1: 'EOF', 2: 'V', 3: 'S', 4: 'DO', 5: 'IO', 6: 'CO', 7: 'Adv', 8: 'Adj', 9: 'T', 10: 'Wy',
                    11: 'WS', 12: 'WE', 13: 'Wi', 14: "Ind"}
    structure = pred.tolist()
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
