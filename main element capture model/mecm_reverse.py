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
                 n_layers=64,
                 dropout=0.2,
                 output_dim=15,
                 bidirectional=False):
        super(MECM, self).__init__()

        # 형태소의 개수가 43개 라서 43개로 함.
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=0)

        # n_layers : 노드 개수
        self.lstm = nn.LSTM(embedding_dim, embedding_dim,
                            num_layers=n_layers, bidirectional=bidirectional, batch_first=True)

        self.hidden_dim = (embedding_dim * 2) if bidirectional else embedding_dim
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        # text = [batch size, sent len]
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        embedded = self.dropout(self.embedding(tokens))

        # embedded = [batch size, sent len, emb dim]
        lstmed, _ = self.lstm(embedded.view(len(tokens), 1, -1))

        # lstmed = [batch size, sent len, hid dim * n directions]
        outputs = self.linear(self.dropout(lstmed.view(len(tokens), -1)))

        # outputs = [batch size, 14]
        scores = F.log_softmax(outputs, dim=1)

        return scores


class StructureDataset(Dataset):
    def __init__(self, csv_file, tokenizer=None, maxsize=512):
        self.datas = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.maxsize = maxsize

        self.dict_main_element = {'PAD': 0, 'EOF': 1, 'V': 2, 'S': 3, 'DO': 4, 'IO': 5, 'CO': 6, 'Adv': 7, 'Adj': 8,
                                  'T': 9, 'Wy': 10, 'WS': 11, 'WE': 12, 'Wi': 13, "Ind": 14}

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        texts = ast.literal_eval(self.datas.iloc[idx, 0]).reverse()
        result = ast.literal_eval(self.datas.iloc[idx, 1]).reverse()

        length = len(texts)
        if length > self.maxsize:
            return

        if self.tokenizer:
            self.tokenizer.tokenizing(texts, justToken=True)
            texts = self.tokenizer.tokens
        result = [self.dict_main_element[char] for char in result]

        # 모든 출력의 길이가 동일해야함
        blank = [0] * (self.maxsize - length)
        texts += blank
        result += blank

        texts = torch.tensor(texts, dtype=torch.long).cuda()
        result = torch.tensor(result, dtype=torch.long).cuda()
        return texts, result, length  # , start, end
