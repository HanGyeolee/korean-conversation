# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 22:57:04 2022

@author: Han
"""

#49152 ~ 65535

import os
import sys
import socket
import subprocess
import json
import dataclass

def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

def launch_entry_console(port):
    if os.name == 'nt': # or use sys.platform for more specific names
        console = ['cmd.exe', '/c'] # or something
    else:
        console = ['xterm', '-e'] # specify your favorite terminal
                                  # emulator here

    cmd = [sys.executable, 'morphemer.py', port]
    return subprocess.Popen(console + cmd)

# 서버 주소
HOST = '127.0.0.1'

# 포트 번호
PORT = get_open_port()
print("server: " + HOST + ':' + str(PORT))

# 소켓 객체를 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 포트 사용중
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print("server: 서버 여는 중")

# bind 함수는 소켓을 특정 네트워크 인터페이스와 포트 번호에 연결하는데 사용됩니다.
# HOST는 hostname, ip address, 빈 문자열 ""이 될 수 있습니다. 
# PORT는 1-65535 사이의 숫자를 사용할 수 있습니다.  
server_socket.bind((HOST,PORT))
print("server: 서버 열림")

# 서버가 클라이언트의 접속을 허용하도록 합니다. 
server_socket.listen()

print("server: 클라이언트 실행 대기")
launch_entry_console(str(PORT))
print("server: 클라이언트 실행 완료")

print("server: 클라이언트 연결 대기")
# accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓을 리턴합니다. 
client_socket, addr = server_socket.accept()

# 접속한 클라이언트의 주소입니다.
print("server: 클라이언트 연결 완료", addr)

# 무한루프를 돌면서 
while True:
    s = input("server: ")
    
    if s == "/exit":
        break;
    elif len(s) > 0:
        # 작성한 문자열을 클라이언트로 전송
        client_socket.sendall(s.encode());
            
        # 클라이언트가 보낸 메시지를 수신하기 위해 대기
        data = client_socket.recv(1024)
        
        response = json.loads(data.decode())
        # 형태소 분리된 데이터
        print("server: 데이터 수신")
        
        result = dataclass.conversation(response["log"], s, response["morpheme"], response["principle"])
        print("server: " + str(result.getDictionary()))

client_socket.sendall("/exit".encode());

# 소켓을 닫습니다.
client_socket.close()
server_socket.close()