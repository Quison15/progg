import pygame
import socket
import threading

# === STEP 1: DISCOVER SERVER ===
def discover_server(broadcast_port=37020):
   # print("[DISCOVERY] Listening for server broadcast...")
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(('', broadcast_port))
    while True:
        data, addr = udp_sock.recvfrom(1024)
        msg = data.decode()
        if msg.startswith("SERVER:"):
            _, ip, port = msg.split(':')
           # print(f"[DISCOVERY] Found server at {ip}:{port}")
            udp_sock.close()
            return ip, int(port)

WIDTH, HEIGHT = 800, 600
CHAT_AREA_HEIGHT = 500
FONT_SIZE = 24

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LAN Chatroom")
font = pygame.font.Font(None, FONT_SIZE)
clock = pygame.time.Clock()

input_text = ''
chat_lines = []

def draw_chat():
    screen.fill((30,30,30))

    y = 10
    for line in chat_lines[-(CHAT_AREA_HEIGHT // FONT_SIZE - 2):]:
        rendered = font.render(line, True, (255,255,255))
        screen.blit(rendered, (10,y))
        y += FONT_SIZE + 4
    
    pygame.draw.rect(screen, (60,60,60), (0,CHAT_AREA_HEIGHT,WIDTH,HEIGHT - CHAT_AREA_HEIGHT))
    input_surface = font.render("> " + input_text, True, (200,200,255))
    screen.blit(input_surface, (10, CHAT_AREA_HEIGHT + 10))
    pygame.display.flip()

def network_listener(sock):
    while True:
        try:
            msg = sock.recv(1024).decode()
            chat_lines.append(msg)
        except:
            break

server_ip, server_port = discover_server()
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((server_ip, server_port))

threading.Thread(target=network_listener, args=(client,), daemon=True).start()

name = input("Enter your name: ")
client.send(name.encode())

running = True
while running:
    draw_chat()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if input_text.lower() == 'quit':
                    running = False
                else:
                    client.send(input_text.encode())
                    input_text = ''
            elif event.key == pygame.K_BACKSPACE:
                input_text = input_text[:-1]
            else:
                if len(input_text) < 100:
                    input_text += event.unicode
    clock.tick(30)

client.close()
pygame.quit()

# # === STEP 2: CONNECT TO SERVER ===
# def start_chat_client(server_ip, server_port):
#     client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client.connect((server_ip, server_port))

#     def receive():
#         while True:
#             try:
#                 msg = client.recv(1024).decode()
#                 if msg:
#                     print(msg)
#                 else:
#                     break
#             except:
#                 break

#     threading.Thread(target=receive, daemon=True).start()

#     name = input("Enter your name: ")
#     client.send(name.encode())

#     while True:
#         msg = input()
#         if msg.lower() == 'quit':
#             break
#         client.send(msg.encode())

#     client.close()

# # === MAIN FLOW ===
# if __name__ == "__main__":
#     ip, port = discover_server()
#     start_chat_client(ip, port)
