import socket
import threading

HOST = "127.0.0.1"
PORT = 12345

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST,PORT))

def receive():
    while True:
        try:
            msg = client.recv(1024).decode()
            print(msg)
        except:
            break

threading.Thread(target=receive).start()

while True:
    msg = input()
    client.send(msg.encode())
    if msg.lower() == 'quit':
        break

client.close()