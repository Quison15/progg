import socket
import threading
import time
import random

TCP_PORT = 12345
UDP_PORT = 37020
MAX_PLAYERS = 6

clients = []
names = []
hands = {}
game_started = False

# === Standard deck ===
suits = ['♠', '♥', '♦', '♣']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
deck = [f"{r}{s}" for s in suits for r in ranks]

def udp_broadcast(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    msg = f"SERVER:{ip}:{port}"
    while not game_started:
        sock.sendto(msg.encode(), ('255.255.255.255', UDP_PORT))
        time.sleep(2)

def broadcast(msg):
    for c in clients:
        try:
            c.sendall(msg.encode())
        except:
            pass

def deal_hands():
    random.shuffle(deck)
    for i, name in enumerate(names):
        hand = [deck.pop(), deck.pop()]
        hands[name] = hand
        try:
            clients[i].sendall(f"Your hand: {hand[0]}, {hand[1]}\n".encode())
        except:
            pass

def handle_client(conn, addr):
    conn.send("Enter your name: ".encode())
    name = conn.recv(1024).decode().strip()
    names.append(name)
    clients.append(conn)
    print(f"[LOBBY] {name} joined from {addr}")
    broadcast(f"{name} has joined the game lobby. ({len(clients)} players)")

    try:
        conn.send("Waiting for game to start...\n".encode())
        while not game_started:
            time.sleep(0.5)

        conn.send("Game is starting!\n".encode())
        time.sleep(1)  # Let them process the start message

        # Once game starts, hands will be dealt

    except:
        print(f"[ERROR] Connection with {name} dropped.")

def wait_for_start():
    global game_started
    while not game_started:
        cmd = input("Type 'start' to begin the game: ")
        if cmd.strip().lower() == 'start':
            game_started = True
            print("[GAME] Game starting...")
            broadcast("Game is starting!\n")
            deal_hands()
            broadcast("[GAME] Hands have been dealt.\n")

def main():
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.bind(('', TCP_PORT))
    tcp_sock.listen()

    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(f"[SERVER] Poker lobby server running on {ip}:{TCP_PORT}")

    threading.Thread(target=udp_broadcast, args=(ip, TCP_PORT), daemon=True).start()
    threading.Thread(target=wait_for_start, daemon=True).start()

    while not game_started:
        conn, addr = tcp_sock.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
        if len(clients) >= MAX_PLAYERS:
            print(f"[LOBBY] Max players reached.")
            break

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
