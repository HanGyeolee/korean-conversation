# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:34:04 2021

@author: Han
"""
from mecab.mecab_tokenizer import Mecab_Tokenizer

tokenizer = Mecab_Tokenizer(dicpath=r'vocab.txt')

string = input()

tokenizer.tokenizing(string)
print(tokenizer.inputstring)
print(tokenizer.morpheme)
print(tokenizer.splited_morpheme)
print(tokenizer.whole)
print(tokenizer.tokens)