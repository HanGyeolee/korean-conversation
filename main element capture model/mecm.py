# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 21:56:25 2022

@author: Han
"""

from __future__ import absolute_import

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

__all__ = ['MECM', 'StructureDataset']

class MECM(nn.Module):
    def __init__(self, 
                 vocab_size=0, 
                 embedding_dim=43, 
                 hidden_dim=8, 
                 n_layers=64, 
                 dropout = 0.25,
                 output_dim=14, 
                 bidirectional=True):
        super(MECM, self).__init__()
        
        # 형태소의 개수가 43개 라서 43개로 함.
        self.embedding = nn.Embedding(vocab_size,
                                       embedding_dim,
                                       padding_idx=0)
        
        # n_layers : 노드 개수
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional = bidirectional)
        
        self.hidden_dim = (hidden_dim*2) if bidirectional else hidden_dim
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens):
        # text = [sent len, batch size] 
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        embedded = self.dropout(self.embedding(tokens))
        
        # embedded = [sent len, batch size, emb dim]
        lstmed, _ = self.lstm(embedded.view(len(tokens), 1, -1))
        
        # output = [sent len, batch size, hid dim * n directions]
        outputs = self.linear(self.dropout(lstmed.view(len(tokens), -1)))
        
        scores = F.log_softmax(outputs, dim=1)
        
        return scores
    
class StructureDataset(Dataset):
    def __init__(self, csv_file, tokenizer=None, maxsize=512): 
        self.datas = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.maxsize = maxsize
        
        self.dict_main_element = {'PAD':0,'EOF':1,'V':2,'S':3,'T':4,'Wy':5,'H':6,'WS':7,'WE':8,'DO':9,'IO':10,'Wi':11,'CO':12,'Adv':13}
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        texts = ast.literal_eval(self.datas.iloc[idx, 0])
        result = ast.literal_eval(self.datas.iloc[idx, 1])
        
        length = len(texts)
        if length > self.maxsize: 
            return
        
        if self.tokenizer:
            self.tokenizer.tokenizing(texts, justToken=True)
            texts = self.tokenizer.tokens
        result = [self.dict_main_element[char] for char in result]    
        
        # 모든 출력의 길이가 동일해야함
        blank = [0]*(self.maxsize - length)
        texts += blank
        result += blank
        
        texts = torch.tensor(texts, dtype=torch.long).cuda()
        result =  torch.tensor(result, dtype=torch.long).cuda()
        return (texts, result, length)#, start, end