# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:34:04 2021

@author: Han
"""
from mecab_custom import Mecab
from mecab_tokenizer import Mecab_Tokenizer

tokenizer = Mecab_Tokenizer(dicpath=r'vocab.txt')
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

string = '경기 성남시 판교신도시에서 이달 분양하는 중대형 아파트의 3.3m²당 분양가가 2006년보다 200만 원 정도 싼 1500만 원 후반대로 결정될 것으로 보인다.'

morpheme = mecab.pos(string, join=True)
print(string)
#print(morpheme)

for index, value in enumerate(morpheme):
    if '+' in value:
        value = value.split('/',1)[1]
        forinsert = value.split('+')
        idx = index
        for insert in forinsert:
            tmp = insert.split('/')
            try:
                if idx == index:
                    morpheme[idx] = tmp[0] + '/' + tmp[1]
                else:
                    morpheme.insert(idx, tmp[0] + '/' + tmp[1])
            except IndexError:
                morpheme.append(tmp[0] + '/' + tmp[1])
            idx += 1
print(morpheme)
            
#vv = [list(result) for result in morpheme if 'VV' in result[1]]

result = [tokenizer.tokenizing(inner) for inner in morpheme]
print(result)