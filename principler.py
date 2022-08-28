# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:15:21 2021

@author: Han
"""
from __future__ import absolute_import

__all__ = ['Principler']

class Principler():
    def __init__(self, verbpath='Verb.csv', elementpath='Element.csv'):
        self.dicpath = verbpath
        try:
            f = open(verbpath, 'r', encoding='utf-8-sig')
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
            
        self.conditionpath = elementpath
        try:
            f = open(elementpath, 'r', encoding='utf-8-sig')
            lines = f.readlines()
            lines = list(map(lambda s: s.strip(), lines)) 
            self.__element = {}
            
            for index, line in enumerate(lines): #list 보다 dictionary의 query 속도가 빠르다.
                s = line.split(',', 1)
                sub = []
                for idx_obj, obj in enumerate(s[1].split(',')):
                    sub.append(obj)
                self.__element[s[0]] = sub
                
            f.close()
        except FileNotFoundError:
            raise Exception('file does not exist at "%s".' % self.dicpath)
            
    def getDict(self):
        return self.__dict
    
    # 문장에서 동사 가져오기
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
    
    # 동사 주요소 가져오기
    def getMainPartbyVerb(self, string):
        v = self.getVerb(string)
        try:
            returns = self.__dict[v]
            returns["V"] = v
            return returns
        except KeyError: # string  값에 따라서 0 혹은 1 이
            #string 에 들어있는 element에 따라 v를 선택한다.
            print("여러 단어 뜻 구분")
            
            base = {'S':'?', 'T':'?', 'Wy':'?', 'H':'*', 'WS':'*', 'WE':'*', 'DO':'*', 'IO':'*', 'Wi':'*', 'CO':'*'}
            for word in string:
                for key, vals in self.__element.items():
                    for val in vals:
                        if word == val:
                            base[key] = '?'
            
            result = 0
            for i in range(0, 5):
                try:
                    if self.CompareDictionary(base, self.__dict[v+str(i)].values()):
                        result = i
                        break
                except:
                    break
            
            returns = self.__dict[v+str(result)]
            returns["V"] = v+str(result)
            return returns 
    
    # 문장에서 주요소 추출하기
    def extractMainPart(self, base, string):
        for idx, word in enumerate(string):
            for key, vals in self.__element.items():
                for val in vals:
                    if (len(val) > 0) & (val in word):
                        if key == "T":
                            base[key] = word
                            string[idx] = ""
                        elif "?" in base[key]:
                            base[key] = string[idx - 1]
                            string[idx - 1] = ""
        
        for idx, word in enumerate(string):
            if ("NNG" in word) | ("NNP" in word) | ("NNB" in word) | ("NNBC" in word) | ("NR" in word) | ("NP" in word):
                if "?" in base['WS']:
                    base['WS'] = word
                    string[idx] = ""
                elif "?" in base['WE']:
                    base['WE'] = word
                    string[idx] = ""
                elif "?" in base['DO']:
                    base['DO'] = word
                    string[idx] = ""
                elif "?" in base['IO']:
                    base['IO'] = word
                    string[idx] = ""
                elif "?" in base['Wi']:
                    base['Wi'] = word
                    string[idx] = ""
                elif "?" in base['CO']:
                    base['CO'] = word
                    string[idx] = ""
        
        return base
         
    def CompareDictionary(self, dict1, list1):
        b = True
        b &= dict1['S'] == list1[0]
        b &= dict1['T'] == list1[1]
        b &= dict1['Wy'] == list1[2]
        b &= dict1['H'] == list1[3]
        b &= dict1['WS'] == list1[4]
        b &= dict1['WE'] == list1[5]
        b &= dict1['DO'] == list1[6]
        b &= dict1['IO'] == list1[7]
        b &= dict1['Wi'] == list1[8]
        b &= dict1['CO'] == list1[9]
        
        return b
        
    def fillData(self, element, splited_morpheme, principle):
        for i, v in enumerate(element):
            if (v == 'V') or (v == 'EOF'):
                continue
            word = splited_morpheme[i].split('/', 1)[0]
            if ('?' not in principle[v]) and ('*' not in principle[v]):
                principle[v] += word
            else:
                principle[v] = word