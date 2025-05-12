import socket
import threading
import time

# === SETTINGS ===
TCP_PORT = 12345
UDP_PORT = 37020
BROADCAST_INTERVAL = 2  # seconds

clients = []
names = []

# === BROADCAST SERVER LOCATION OVER UDP ===
def udp_broadcast(ip, tcp_port):
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    message = f"SERVER:{ip}:{tcp_port}"
    print(f"[BROADCAST] Broadcasting as {message}")

    while True:
        udp_sock.sendto(message.encode(), ('255.255.255.255', UDP_PORT))
        time.sleep(BROADCAST_INTERVAL)

# === SEND MESSAGE TO ALL CLIENTS EXCEPT OPTIONAL SENDER ===
def broadcast(msg, sender=None):
    for client in clients:
        if client != sender:
            try:
                client.sendall(msg.encode())
            except:
                clients.remove(client)

# === HANDLE EACH CLIENT ===
def handle_client(conn, addr):
    try:
        conn.send("Your name: ".encode())
        name = conn.recv(1024).decode()
        names.append(name)
        clients.append(conn)
        print(f"[JOIN] {name} joined from {addr}")
        broadcast(f"{name} has joined the chat.", conn)

        while True:
            msg = conn.recv(1024).decode()
            if msg.lower() == 'quit':
                break
            broadcast(f"{name}: {msg}", conn)

    except:
        pass
    finally:
        if conn in clients:
            clients.remove(conn)
        if name in names:
            names.remove(name)
        conn.close()
        broadcast(f"{name} has left the chat.")
        print(f"[LEAVE] {name} disconnected")

# === MAIN SERVER LOOP ===
def main():
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.bind(('', TCP_PORT))  # bind to all interfaces
    tcp_sock.listen()

    # Get local IP to broadcast
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # Start UDP broadcaster thread
    threading.Thread(target=udp_broadcast, args=(local_ip, TCP_PORT), daemon=True).start()

    print(f"[SERVER] Chat server started on {local_ip}:{TCP_PORT}")
    while True:
        conn, addr = tcp_sock.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()
