# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 21:06:14 2021

@author: Han
"""
from __future__ import absolute_import

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

__all__ = ['StructrueDataset', 'LSTMTagger']

class StructrueDataset(Dataset):
    def __init__(self, csv_file, tokenizer=None, maxsize=512): 
        self.datas = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.maxsize = maxsize
        
        self.dict_main_element = {'EOF':0,'V':1,'S':2,'T':3,'Wy':4,'WS':5,'WE':6,'DO':7,'IO':8,'H':9,'Wi':10,'CO':11}
    
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
        #start = ast.literal_eval(self.datas.iloc[idx, 2])
        #end = ast.literal_eval(self.datas.iloc[idx, 3])
        
        if self.tokenizer:
            self.tokenizer.tokenizing(texts, justToken=True)
            texts = self.tokenizer.tokens
        result = [self.dict_main_element[char] for char in result]      
        
        texts = torch.tensor(texts, dtype=torch.long).cuda()
        result =  torch.tensor(result, dtype=torch.long).cuda()
        #start = torch.IntTensor(start).cuda()
        #end = torch.IntTensor(end).cuda()
        return (texts, result)#, start, end
        
class LSTMTagger(nn.Module):
    def __init__(self, vocab_size=0, embedding_dim=20, hidden_dim=8, output_dim=12, n_layers=64, bidirectional=True, dropout = 0.25):
        super(LSTMTagger, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional = bidirectional)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):        
        # text = [sent len, batch size] 
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.lstm(embedded.view(len(text), 1, -1)) 

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        tag_space = self.fc(self.dropout(outputs.view(len(text), -1))) 

        predictions = F.log_softmax(tag_space, dim = 1)
        
        return predictions