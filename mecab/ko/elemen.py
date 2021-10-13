# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:06:44 2021

@author: Han
"""
import torch

class ELEMENTagger():
    """Element Tagger with Mecab.
    
    """
    def __init__(self, ptpath='model_result.pt'):
        self.model_result = torch.load(ptpath)
        self.dict_element = {1:'EOF',2:'V',3:'S',4:'T',5:'Wy',6:'WS',7:'WE',8:'DO',9:'IO',10:'H',11:'Wi'}
 
    def getElement(self, tokens):
        """Element Tagger with Mecab.
        
        :param tokens: token from mecab_tokenizer
        """
        prediction = self.model_result(torch.tensor(tokens, dtype=torch.long))
        
        structure = prediction.tolist()
        for i, sub in enumerate(structure):
            structure[i] = sub.index(max(sub))
            
        return [self.dict_element[v] for v in structure]