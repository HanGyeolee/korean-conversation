# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:34:04 2021

@author: Han
"""
import datetime
now = datetime.datetime.now()

string = input()

from mecab.mecab_tokenizer import Mecab_Tokenizer


tokenizer = Mecab_Tokenizer(dicpath=r'vocab.txt', update=False)

tokenizer.spliter(string, allattrs=True)
#print(tokenizer.inputstring)
print(tokenizer.morpheme)
print(tokenizer.splited_morpheme)
#print(tokenizer.whole)
#print(tokenizer.tokens)
print(now.strftime('%Y.%m.%d.%H:%M:%S.%f'))


"""
from konlpy.tag import Okt

okt = Okt()
print(okt.phrases(string))
print(okt.pos(string, norm=True))
"""