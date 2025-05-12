import socket
import threading

def discover_server(port=37020):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', port))
    while True:
        data, _ = sock.recvfrom(1024)
        msg = data.decode()
        if msg.startswith("SERVER:"):
            _, ip, port = msg.split(':')
            return ip, int(port)

def receive(sock):
    while True:
        try:
            msg = sock.recv(1024).decode()
            if msg:
                print(msg)
        except:
            break

ip, port = discover_server()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ip, port))

threading.Thread(target=receive, args=(sock,), daemon=True).start()

name = input("Enter your name: ")
sock.send(name.encode())

while True:
    try:
        pass  # just wait for messages from server
    except KeyboardInterrupt:
        break

sock.close()
