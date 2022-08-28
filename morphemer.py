# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 00:37:16 2022

@author: Han
"""

import sys
import socket
import datetime
import mecab.ko as Mko
import principler
import json

print("client: 클라이언트 실행")
HOST = '127.0.0.1'  
PORT = int(sys.argv[1])

print("client: 인공지능 준비")
tokenizer = Mko.Tokenizer(dicpath=r'vocab.txt', update=False)
principler = principler.Principler(verbpath=u"mecab/ko/Verb.csv", elementpath=u"mecab/ko/Element.csv")
print("client: 인공지능 준비 완료")

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("client: 서버 연결 대기")
client_socket.connect((HOST, PORT))
print("client: 서버 연결 완료\n")

copy = '0'

while True:
    # 서버가 보낸 메시지를 수신하기 위해 대기
    data = client_socket.recv(1024)

    now = datetime.datetime.now()
    
    print("client: 데이터 수신")
    if data.decode() == "/exit":
        break;        
    copy = data
    
    tokenizer.tokenizing(data.decode(), allattrs=False)
    principle = principler.getMainPartbyVerb(tokenizer.splited_morpheme)
    
    result = {}
    result["log"] = now.strftime('%Y.%m.%d.%w.%H:%M:%S.%f')
    result["morpheme"] = tokenizer.splited_morpheme
    result["principle"] = principle
    
    print("client: 데이터 송신")
                 
    # 형태소로 분할된 메시지 송신
    client_socket.sendall(json.dumps(result, ensure_ascii=False).encode())
    
# 소켓을 닫습니다.
client_socket.close()
