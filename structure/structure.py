# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:37:54 2021

@author: Han
"""
import sys
sys.path.append("..")

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, Dataset, DataLoader
import time
import random

from mecab.mecab_tokenizer import Mecab_Tokenizer

#Geforce 3070 need >cuda11.1

class StructrueDataset(Dataset):
    def __init__(self, csv_file, tokenizer=None, maxsize=512): 
        self.datas = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.maxsize = maxsize
        
        self.dict_main_element = {'EOF':1,'V':2,'S':3,'T':4,'Wy':5,'WS':6,'WE':7,'DO':8,'IO':9,'H':10,'Wi':11}
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        length = int(self.datas.iloc[idx, 4])
        if length > self.maxsize: 
            return
        
        texts = ast.literal_eval(self.datas.iloc[idx, 0])
        result = ast.literal_eval(self.datas.iloc[idx, 1])
        start = ast.literal_eval(self.datas.iloc[idx, 2])
        end = ast.literal_eval(self.datas.iloc[idx, 3])
        if self.tokenizer:
            self.tokenizer.tokenizing(texts, justToken=True)
            texts = self.tokenizer.tokens
        result = [self.dict_main_element[char] for char in result]
        
        blank = [0]*(self.maxsize - length)
        
        texts += blank
        result += blank
        start += blank
        end += blank
        
        result = np.eye(12)[result]
        
        texts = torch.IntTensor(texts).cuda()
        result = torch.LongTensor(result).cuda()
        start = torch.IntTensor(start).cuda()
        end = torch.IntTensor(end).cuda()
        return texts, result, start, end
        
class RNNSTRUCTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, batch_size):
        super(RNNSTRUCTagger, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional = bidirectional)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden = torch.zeros(1, batch_size, hidden_dim, requires_grad=True)
        self.cell = torch.zeros(1, batch_size, hidden_dim, requires_grad=True)
        
    def forward(self, text):
        text = torch.transpose(text, 0, 1)
        
        # text = [sent len, batch size] 
        # https://stackoverflow.com/questions/62081155/pytorch-indexerror-index-out-of-range-in-self-how-to-solve
        embedded = self.dropout(self.embedding(text)) # embedded [MAX_SEN, 5, EMBEDDING_DIM]

        # embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.lstm(embedded) # outputs[MAX_SEN, 5, HIDDEN_DIM * 2]

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        predictions = self.fc(self.dropout(outputs))  # predictions [MAX_SEN, 5, OUTPUT_DIM]

        # predictions = [sent len, batch size, output dim]
        return predictions

tokenizer = Mecab_Tokenizer(dicpath=r'../vocab.txt', update=False)

#랜덤 시드 고정
SEED = 20211012
random.seed(SEED)
torch.manual_seed(SEED)

MAX_SEN = 256
INPUT_DIM = tokenizer.getMax() + 1
OUTPUT_DIM = 12
EMBEDDING_DIM = 20
HIDDEN_DIM = 128
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.25
BATCH_SIZE = 3

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = StructrueDataset(maxsize=MAX_SEN, csv_file='stru_train_dataset.csv', tokenizer=tokenizer)
#valid_data = StructrueDataset(maxsize=MAX_SEN, csv_file='stru_valid_dataset.csv', tokenizer=tokenizer)
test_data = StructrueDataset(maxsize=MAX_SEN, csv_file='stru_train_dataset.csv', tokenizer=tokenizer)


train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
#valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

model_result = RNNSTRUCTagger(INPUT_DIM, 
                     EMBEDDING_DIM, 
                     HIDDEN_DIM, 
                     OUTPUT_DIM, 
                     N_LAYERS, 
                     BIDIRECTIONAL, 
                     DROPOUT,
                     BATCH_SIZE).cuda()

model_result.embedding.weight.data[0] = torch.zeros(EMBEDDING_DIM)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model_result):,} trainable parameters')
optimizer = optim.Adam(model_result.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

"""
tmpppp = torch.rand(64, 1, 512).cuda()
prediction = model(tmpppp)
print(prediction)
"""

def train(epoch):
    model_result.train()
    for batch_idx, batch in enumerate(train_dataloader): # batch (variable, 4, 512)
        optimizer.zero_grad()
        
        prediction = model_result(batch[0])
        batch[1] = torch.transpose(batch[1], 0, 1)
        
        #print("prediction",prediction.shape)
        #print("result",batch[1][:, -1].shape)
        #a = prediction.tolist()
        #b = batch[1][:, 0].tolist()
        
        loss = criterion(prediction, batch[1][:, 0])
        loss.backward()
        
        optimizer.step()
        
        if batch_idx%50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(batch[0]),
                len(train_dataloader.dataset),100.
                * batch_idx / len(train_dataloader), 
                loss.data))
            
def test():
    model_result.eval()
    test_loss = 0
    correct = 0
    for batch in test_dataloader: # batch (variable, 4, 512)
        prediction = model_result(batch[0])
        
        batch[1] = torch.transpose(batch[1], 0, 1)
        
        test_loss += criterion(prediction, batch[1][:, 0]).data[0]
        
        pred = prediction.data.max(1,keepdim=True)[1]
        correct += pred.eq(batch[1][:, 0].data.view_as(pred)).sum()
    
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

for epoch in range(1, 1000):
    train(epoch)



