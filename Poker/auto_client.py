import socket
import threading

# === STEP 1: DISCOVER SERVER ===
def discover_server(broadcast_port=37020):
    print("[DISCOVERY] Listening for server broadcast...")
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(('', broadcast_port))
    while True:
        data, addr = udp_sock.recvfrom(1024)
        msg = data.decode()
        if msg.startswith("SERVER:"):
            _, ip, port = msg.split(':')
            print(f"[DISCOVERY] Found server at {ip}:{port}")
            udp_sock.close()
            return ip, int(port)

# === STEP 2: CONNECT TO SERVER ===
def start_chat_client(server_ip, server_port):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))

    def receive():
        while True:
            try:
                msg = client.recv(1024).decode()
                if msg:
                    print(msg)
                else:
                    break
            except:
                break

    threading.Thread(target=receive, daemon=True).start()

    name = input("Enter your name: ")
    client.send(name.encode())

    while True:
        msg = input()
        if msg.lower() == 'quit':
            break
        client.send(msg.encode())

    client.close()

# === MAIN FLOW ===
if __name__ == "__main__":
    ip, port = discover_server()
    start_chat_client(ip, port)
