# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:34:04 2021

@author: Han
"""

string = input()

from mecab.mecab_tokenizer import Mecab_Tokenizer

tokenizer = Mecab_Tokenizer(dicpath=r'vocab.txt')

tokenizer.tokenizing(string, allattrs=True)
print(tokenizer.inputstring)
print(tokenizer.morpheme)
print(tokenizer.splited_morpheme)
print(tokenizer.whole)
print(tokenizer.tokens)


from konlpy.tag import Okt

okt = Okt()
print(okt.phrases(string))
print(okt.pos(string, norm=True))