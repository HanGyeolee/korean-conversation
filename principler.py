# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:15:21 2021

@author: Han
"""
from __future__ import absolute_import

__all__ = ['Principler']

class Principler():
    def __init__(self, verbpath='Verb.csv'):
        self.dicpath = verbpath
        try:
            f = open(verbpath, 'r', encoding='utf-8')
            lines = f.readlines()
            lines = list(map(lambda s: s.strip(), lines)) 
            self.__dict = {}
            
            sub = []
            for index, line in enumerate(lines): #list 보다 dictionary의 query 속도가 빠르다.
                if index == 0:
                    sub = line.split(',', 1)[1].split(',')
                else:
                    s = line.split(',', 1)
                    d = {}
                    for idx_obj, obj in enumerate(s[1].split(',')):
                        d[sub[idx_obj]] = obj
                    self.__dict[s[0]] = d
                
            f.close()
        except FileNotFoundError:
            raise Exception('file does not exist at "%s".' % self.dicpath)
            
    def getDict(self):
        return self.__dict
    
    def getVerb(self, morpheme):
        #try:
        for v in reversed(morpheme):
            if 'VV' in v:
                return v
        for v in reversed(morpheme):
            if 'VA' in v:
                return v
        for v in reversed(morpheme):
            if 'V' in v:
                return v
        #except:
        #    pass
        return ''
    
    def setVerb(self, string):
        v = self.getVerb(string)
        try:
            return self.__dict[v]
        except KeyError: # string  값에 따라서 0 혹은 1 이
            return self.__dict[v+"0"]
        
    def fillData(self, element, splited_morpheme, principle):
        for i, v in enumerate(element):
            if (v == 'V') or (v == 'EOF'):
                continue
            word = splited_morpheme[i].split('/', 1)[0]
            if ('?' not in principle[v]) and ('*' not in principle[v]):
                principle[v] += word
            else:
                principle[v] = word