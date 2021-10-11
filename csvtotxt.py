# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 02:44:24 2021

@author: Han
"""
import os
import csv

def ChangeToTxt(dicpath, destpath):
    f = open(dicpath, 'r', encoding='utf-8')
    ff = open(destpath, 'a', encoding='utf-8')
    lines = csv.reader(f)
    for line in lines:
        ff.write(line[0]+'/'+line[4] +'\n')
    f.close()
    ff.close()
    
path = r'C:\mecab\mecab-ko-dic'
filedirs = os.listdir(path)
filedirs_csv = [file for file in filedirs if file.endswith(".csv")]

"""
b = [a for a in filedirs_csv if 'VV' in a][0]
b = path+'\\'+b

ChangeToTxt(b, r'D:\ProjectFiles\python-project\MecabTest')
"""
for file in filedirs_csv :
    file = path + '\\' + file
    ChangeToTxt(file, r'D:\ProjectFiles\python-project\MecabTest\vocab.txt')