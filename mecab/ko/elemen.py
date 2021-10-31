# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:06:44 2021

@author: Han
"""
from __future__ import absolute_import

import torch
try:
    from tagger import LSTMTagger
except ImportError:
    from .tagger import LSTMTagger
    pass

__all__ = ['ELEMENTagger']

class ELEMENTagger():
    """Element Tagger with Mecab.
    
    """
    def __init__(self, vocab_size=655360, ptpath='model_result.pt', maxsize=512):
        device = torch.device('cpu')
        self.maxsize = maxsize
        self.model_result = LSTMTagger(vocab_size=vocab_size)
        checkpoint = torch.load(ptpath, map_location=device)
        self.model_result.load_state_dict(checkpoint['model_state_dict'])
        self.dict_element = {0:'', 1:'EOF',2:'V',3:'S',4:'T',5:'Wy',6:'WS',7:'WE',8:'DO',9:'IO',10:'H',11:'Wi',12:'CO'}
 
    def getElement(self, tokens):
        """Element Tagger with Mecab.
        
        :param tokens: token from mecab_tokenizer
        """        
        prediction = self.model_result(torch.tensor(tokens, dtype=torch.long))
        
        structure = prediction.tolist()
        for i, sub in enumerate(structure):
            structure[i] = sub.index(max(sub))
            
        return [self.dict_element[v] for v in structure]