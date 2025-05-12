import socket
import threading

HOST = '127.0.0.1'
PORT = 12345

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST,PORT))
server.listen()

clients = []
names = []

def broadcast(msg, sender_conn=None):
    for client in clients:
        if client != sender_conn:
            try:
                client.sendall(msg.encode())
            except:
                clients.remove(client)

def handle_client(conn, addr):
    conn.send("Your name: ".encode())
    name = conn.recv(1024).decode()
    names.append(name)
    clients.append(conn)

    print(f"{name} joined from {addr}")
    broadcast(f"{name} joined the chat!", conn)

    while True:
        try:
            msg = conn.recv(1024).decode()
            if msg.lower() == 'quit':
                break
            broadcast(f"{name}: {msg}", conn)
        except:
            break
    conn.close()
    clients.remove(conn)
    names.remove(name)
    broadcast(f"{name} left the chat.")

while True:
    conn, addr = server.accept()
    threading.Thread(target=handle_client, args=(conn,addr)).start()