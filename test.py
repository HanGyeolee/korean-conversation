# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:34:04 2021

@author: Han
"""
import datetime
import mecab.ko as Mko

now = datetime.datetime.now()

string = input()
print(now.strftime('%Y.%m.%d.%w.%H:%M:%S.%f'))

tokenizer = Mko.Tokenizer(dicpath=r'vocab.txt', update=False)

tokenizer.tokenizing(string, allattrs=False)
#tokenizer.spliter(string, allattrs=True)
#print(tokenizer.inputstring)
#print(tokenizer.morpheme)
print(tokenizer.splited_morpheme)
print(tokenizer.whole)
#print(tokenizer.tokens)

elementagger = Mko.ELEMENTagger(ptpath=u"mecab/ko/model_result.pt", vocab_size=tokenizer.getMax() + 1)
#print(elementagger.getElement(tokenizer.tokens))

"""
from konlpy.tag import Okt

okt = Okt()
print(okt.phrases(string))
print(okt.pos(string, norm=True))
"""