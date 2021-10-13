# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:56:09 2021

@author: Han
"""
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
import random

from tokenizer import Tokenizer

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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
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

tokenizer = Tokenizer(dicpath=r'../../vocab.txt', update=False)

#랜덤 시드 고정
SEED = 20211012
random.seed(SEED)
torch.manual_seed(SEED)

MAX_SEN = 512
INPUT_DIM = tokenizer.getMax() + 1
OUTPUT_DIM = 12
EMBEDDING_DIM = 20
HIDDEN_DIM = 6
N_LAYERS = 64
BIDIRECTIONAL = True
DROPOUT = 0.25

train_data = StructrueDataset(maxsize=MAX_SEN, csv_file='elemen_train_dataset.csv', tokenizer=tokenizer)
#valid_data = StructrueDataset(maxsize=MAX_SEN, csv_file='stru_valid_dataset.csv', tokenizer=tokenizer)
test_data = StructrueDataset(maxsize=MAX_SEN, csv_file='elemen_train_dataset.csv', tokenizer=tokenizer)

model_result = LSTMTagger(INPUT_DIM, 
                     EMBEDDING_DIM, 
                     HIDDEN_DIM, 
                     OUTPUT_DIM, 
                     N_LAYERS, 
                     BIDIRECTIONAL, 
                     DROPOUT).cuda()

model_result.embedding.weight.data[0] = torch.zeros(EMBEDDING_DIM).cuda()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f'The model has {count_parameters(model_result):,} trainable parameters')
optimizer = optim.Adam(model_result.parameters())
criterion = nn.NLLLoss()

def train(epoch):
    model_result.train()
    for batch_idx, (sentences, tags) in enumerate(train_data):
        optimizer.zero_grad()
        
        prediction = model_result(sentences)
        
        loss = criterion(prediction, tags)
        loss.backward()
        optimizer.step()
        
        if batch_idx%5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx ,
                len(train_data),
                100. * batch_idx / len(train_data), 
                loss.data))
            
def test():
    model_result.eval()
    test_loss = 0
    correct = 0
    for sentences, tags in test_data: # batch (variable, 4, 512)
        prediction = model_result(sentences)
        
        test_loss += criterion(prediction, tags).data[0]
        
        pred = prediction.data.max(1,keepdim=True)[1]
        correct += pred.eq(tags.data.view_as(pred)).sum()
    
    test_loss /= len(test_data)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))

PATH = "model_result.pt"
if True:
    for epoch in range(1000):
        train(epoch)
    
    torch.save(model_result, PATH)
else:
    model_result = torch.load(PATH)

prediction = model_result(train_data[0][0])

def getstructure(prediction):
    dict_element = {1:'EOF',2:'V',3:'S',4:'T',5:'Wy',6:'WS',7:'WE',8:'DO',9:'IO',10:'H',11:'Wi'}
    structure = prediction.tolist()
    for i, sub in enumerate(structure):
        structure[i] = sub.index(max(sub))
        
    print(structure)
    print(train_data[0][1])
    return [dict_element[v] for v in structure]
    
print(getstructure(prediction))
    
    
    
    
    