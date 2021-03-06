# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:34:04 2021

@author: Han
"""
import datetime
import mecab.ko as Mko
import principler

now = datetime.datetime.now()

print("입력:")
string = input()
print(now.strftime('%Y.%m.%d.%w.%H:%M:%S.%f'))

# 필요 클래스 선언
tokenizer = Mko.Tokenizer(dicpath=r'vocab.txt', update=False)
principler = principler.Principler(verbpath=u"mecab/ko/Verb.csv", elementpath=u"mecab/ko/Element.csv")
#elementagger = Mko.ELEMENTagger(ptpath=u"mecab/ko/model_result.pt")

# 토크나이징
tokenizer.tokenizing(string, allattrs=False)
#tokenizer.spliter(string, allattrs=True)
#print(tokenizer.inputstring)
#print(tokenizer.morpheme)
print(tokenizer.splited_morpheme)
#print(tokenizer.whole)
#print(tokenizer.tokens)

# 동사를 이용한 정보 배제
verb, principle = principler.getMainPartbyVerb(tokenizer.splited_morpheme)
principler.extractMainPart(principle, tokenizer.splited_morpheme)

# 주요소 태깅
#element = elementagger.getElement(tokenizer.tokens)
#print()

# 태깅 정보 저장
#principler.fillData(element, tokenizer.splited_morpheme, principle)

#print(element)
print(verb + ":" + str(principle))
"""
from konlpy.tag import Okt

okt = Okt()
print(okt.phrases(string))
print(okt.pos(string, norm=True))
"""