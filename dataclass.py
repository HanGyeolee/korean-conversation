# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 01:55:47 2022

@author: Han
"""
from __future__ import absolute_import

__all__ = ['conversation']

class conversation:
    def __init__(self, time, string, morpheme, principle):
        self.log = time;
        self.raw = string;
        self.morpheme = morpheme;
        self.reversed = list(reversed(morpheme));
        self.sentencetype = "";
        self.predicate = predicate(principle);
        
        self.predicate.morphemeAnalysis(morpheme);
    
    def getDictionary(self):
        return { 
            "log": self.log,
            "raw": self.raw,
            "morpheme": self.morpheme,
            "reversed": self.reversed,
            "sentencetype": self.sentencetype,
            "predicate": self.predicate.getDictionary()
        }
        
class predicate:
    def __init__(self, principle):
        self.tense = "";
        self.adj = "";
        self.type = "";
        self.word = principle["V"];
        self.adv = [];
        self.complement = principle["CO"];
        self.with_ = principle["Wi"];
        self.indirectobject = principle["IO"];
        self.directobject = principle["DO"];
        self.whereend = principle["WE"];
        self.wherestart = principle["WS"];
        self.how = principle["H"];
        self.why = principle["Wy"];
        self.time = principle["T"];
        self.subject = principle["S"];
        
    # 형태소 분석
    def morphemeAnalysis(self, morpheme):
        return 
        
    def getDictionary(self):
        return {
            "tense":        self.tense,
            "adj":          self.adj,
            "type":         self.type,
            "word":         self.word,
            "adv":          self.adv,
            "complement":   self.complement,
            "with":         self.with_,
            "indirectobject": self.indirectobject,
            "directobject": self.directobject,
            "whereend":     self.whereend,
            "wherestart":   self.wherestart,
            "how":          self.how,
            "why":          self.why,
            "time":         self.time,
            "subject":      self.subject
        }
        
class etc:
    def __init__(self, word):
        self.__dict = {
            "type":         "",
            "word":         word,
            "adv":          []
        }