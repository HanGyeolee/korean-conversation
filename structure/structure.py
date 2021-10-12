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
from torch.utils.data import Dataset, DataLoader
import time
import random

from mecab.mecab_tokenizer import Mecab_Tokenizer

#Geforce 3070 need >cuda11.1

class StructrueDataset(Dataset):
    def __init__(self, csv_file, tokenizer=None):
        self.datas = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.dict_main_element = {'EOF':1,'V':2,'S':3,'T':4,'Wy':5,'WS':6,'WE':7,'DO':8,'IO':9,'H':10,'Wi':11}
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx, maxsize=512):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        length = int(self.datas.iloc[idx, 4])
        if length>512: return
        
        texts = ast.literal_eval(self.datas.iloc[idx, 0])
        result = ast.literal_eval(self.datas.iloc[idx, 1])
        start = ast.literal_eval(self.datas.iloc[idx, 2])
        end = ast.literal_eval(self.datas.iloc[idx, 3])
        sample = {'text': texts, 'result': result, 'start':start,'end':end }
        if self.tokenizer:
            self.tokenizer.tokenizing(sample['text'], justToken=True)
            sample['text'] = self.tokenizer.tokens
            sample['result'] = [self.dict_main_element[char] for char in sample['result']]
            
        blank = [0]*(512-length)
        
        sample['text'] += blank
        sample['result'] += blank
        sample['start'] += blank
        sample['end'] += blank
            
        sample['text'] = torch.Tensor([sample['text']])
        sample['result'] = torch.Tensor([sample['result']])
        sample['start'] = torch.Tensor([sample['start']])
        sample['end'] = torch.Tensor([sample['end']])
        return sample

class RNNSTRUCTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional = bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]
        return predictions

#랜덤 시드 고정
SEED = 20211012
random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIM = INPUT_DIM = 512
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = Mecab_Tokenizer(dicpath=r'../vocab.txt', update=False)

train_data = StructrueDataset(csv_file='stru_train_dataset.csv', tokenizer=tokenizer)
#valid_data = StructrueDataset(csv_file='stru_valid_dataset.csv', tokenizer=tokenizer)
#test_data = StructrueDataset(csv_file='stru_test_dataset.csv', tokenizer=tokenizer)

dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

model = RNNSTRUCTagger(INPUT_DIM, 
                     EMBEDDING_DIM, 
                     HIDDEN_DIM, 
                     OUTPUT_DIM, 
                     N_LAYERS, 
                     BIDIRECTIONAL, 
                     DROPOUT)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=[765795, 765792, 765788])

model = model.to(device)
criterion = criterion.to(device)

prediction = model(dataloader[0]['text'])
print(prediction.shape)




