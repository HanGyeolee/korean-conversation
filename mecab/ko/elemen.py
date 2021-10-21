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
    def __init__(self, vocab_size=655360, ptpath='model_result.pt'):
        device = torch.device('cpu')
        self.model_result = LSTMTagger(vocab_size=vocab_size)
        self.model_result.load_state_dict(torch.load(ptpath, map_location=device))
        self.dict_element = {0:'EOF',1:'V',2:'S',3:'T',4:'Wy',5:'WS',6:'WE',7:'DO',8:'IO',9:'H',10:'Wi',11:'CO'}
 
    def getElement(self, tokens):
        """Element Tagger with Mecab.
        
        :param tokens: token from mecab_tokenizer
        """
        prediction = self.model_result(torch.tensor(tokens, dtype=torch.long))
        
        structure = prediction.tolist()
        for i, sub in enumerate(structure):
            structure[i] = sub.index(max(sub))
            
        return [self.dict_element[v] for v in structure]