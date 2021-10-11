# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:22:19 2021

@author: Han
"""

class Mecab_Tokenizer():
    """Tokenizer with Mecab.
    """
    def __init__(self, dicpath=''):
        self.dicpath = dicpath
        try:
            f = open(dicpath, 'r', encoding='utf-8')
            lines = f.readlines()
            lines = list(map(lambda s: s.strip(), lines)) 
            self.__dict = {}
            
            for index, line in enumerate(lines): #list 보다 dictionary의 query 속도가 빠르다.
                self.__dict[line] = index
                self.__index = index
                
            f.close()
        except FileNotFoundError:
            raise Exception('file does not exist at "%s".' % dicpath)
            
    def tokenizing(self, v):
        """Tokenizer.

        :param v: vocab.
        """
        try:
            tmp = int(v.split("/")[0])
            result = self.__dict["${number}/SN"] # 숫자는 다 묶어서 같은 토큰으로 친다.
            return [v, result]
        except ValueError:
            try:
                result = self.__dict[v]
                return [v, result]
            except :
                if False & ('ㅓ/EC' not in v):
                    f = open(self.dicpath, 'a', encoding='utf-8')
                    f.write(v +'\n')
                    f.close()
                    
                    self.__index += 1
                    self.__dict[v] = self.__index
                        
                    return self.__index
                return [v, -1]