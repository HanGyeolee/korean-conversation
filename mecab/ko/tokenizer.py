# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:22:19 2021

@author: Han
"""
from __future__ import absolute_import

try:
    from mecab_custom import Mecab
except ImportError:
    from .mecab_custom import Mecab
    pass
    
__all__ = ['Tokenizer']

class Tokenizer():
    """ElementTokenizer with Mecab.
    
    inputstring: 입력된 문장    
    morpheme: 형태소     
    info: 정보    
    token: 토크나이즈된 문장
    """
    def __init__(self, dicpath='', update=False):
        self.dicpath = dicpath
        self.update = update
        self.__mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
        try:
            f = open(dicpath, 'r', encoding='utf-8')
            lines = f.readlines()
            lines = list(map(lambda s: s.strip(), lines)) 
            self.__dict = {}
            
            for index, line in enumerate(lines): #list 보다 dictionary의 query 속도가 빠르다.
                self.__dict[line] = index+1
                self.__index = index+1
                
            f.close()
        except FileNotFoundError:
            raise Exception('file does not exist at "%s".' % dicpath)
            
    def getMax(self):
        return self.__index
    
    def getDict(self):
        return self.__dict
            
    def __match(self, v):
        """Tokenizer.

        :param v: vocab.
        """
        try:
            int(v.split("/")[0])
            return [v, self.__dict["${number}/SN"]] # 숫자는 다 묶어서 같은 토큰으로 친다.
        except ValueError:
            try:
                if 'VV' in v: # 동사도 다 묶어서 같은 토큰으로 친다.
                    return [v, self.__dict["${verb}/VV"]]
                elif 'VA' in v: # 형용사도 다 묶어서 같은 토큰으로 친다.
                    return [v, self.__dict["${adjective}/VA"]]
                elif 'SF' in v: # 종결문자도 다 묶어서 같은 토큰으로 친다.
                    return [v, self.__dict["${mark}/SF"]]
                elif 'EF' in v: # 종결어미도 다 묶어서 같은 토큰으로 친다.
                    return [v, self.__dict["${end}/EF"]]
                elif 'SL' in v: # 영어도 다 묶어서 같은 토큰으로 친다.
                    return [v, self.__dict["${english}/SL"]]
                elif 'SY' in v: # 특수문자도 다 묶어서 같은 토큰으로 친다.
                    return [v, self.__dict["${special}/SY"]]
                return [v, self.__dict[v]]
            except :
                if self.update & ('ㅓ/EC' not in v):
                    f = open(self.dicpath, 'a', encoding='utf-8')
                    f.write(v +'\n')
                    f.close()
                    
                    self.__index += 1
                    self.__dict[v] = self.__index
                        
                    return [v, self.__index]
                return [v, -1]
    def spliter(self, string, allattrs=False):
        if string != "":
            self.morpheme = self.__mecab.pos(string, join=True, allattrs=allattrs)
            self.splited_morpheme = self.__mecab.pos(string, join=True)
            self.inputstring = string
            
            for index, value in enumerate(self.splited_morpheme):
                if '+' in value:
                    value = value.split('/',1)[1]
                    forinsert = value.split('+')
                    idx = index
                    for insert in forinsert:
                        tmp = insert.split('/')
                        try:
                            if idx == index:
                                self.splited_morpheme[idx] = tmp[0] + '/' + tmp[1]
                            else:
                                self.splited_morpheme.insert(idx, tmp[0] + '/' + tmp[1])
                        except IndexError:
                            self.splited_morpheme.append(tmp[0] + '/' + tmp[1])
                        idx += 1
        else:
            print("입력이 없습니다.")
        
    def tokenizing(self, string, allattrs=False, justToken=False):
        if string != "":
            if justToken:
                self.splited_morpheme = string
            else:
                self.spliter(string, allattrs)
                
            self.whole = [self.__match(inner) for inner in self.splited_morpheme]
            self.tokens = [data[1] for data in self.whole]
        else:
            print("입력이 없습니다.")