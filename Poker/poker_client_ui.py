import pygame
import socket
import threading

# === Settings ===
WIDTH, HEIGHT = 800, 600
CHAT_HEIGHT = 500
FONT_SIZE = 24

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poker Lobby")
font = pygame.font.Font(None, FONT_SIZE)
clock = pygame.time.Clock()

input_text = ''
chat_lines = []

# === UDP Discovery ===
def discover_server(port=37020):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', port))
    while True:
        data, _ = sock.recvfrom(1024)
        msg = data.decode()
        if msg.startswith("SERVER:"):
            _, ip, port = msg.split(':')
            sock.close()
            return ip, int(port)

# === Display UI ===
def draw_ui():
    screen.fill((20, 20, 20))
    
    # Chat log
    y = 10
    for line in chat_lines[-(CHAT_HEIGHT // FONT_SIZE - 2):]:
        txt = font.render(line, True, (255, 255, 255))
        screen.blit(txt, (10, y))
        y += FONT_SIZE + 4

    # Input area
    pygame.draw.rect(screen, (40, 40, 40), (0, CHAT_HEIGHT, WIDTH, HEIGHT - CHAT_HEIGHT))
    input_surface = font.render("> " + input_text, True, (200, 200, 255))
    screen.blit(input_surface, (10, CHAT_HEIGHT + 10))

    pygame.display.flip()

# === Receive messages from server ===
# === inside receive_messages(sock)
def receive_messages(sock):
    while True:
        try:
            msg = sock.recv(1024).decode()
            if msg:
                chat_lines.append(msg.strip())
                # Optional: highlight hand
                if msg.startswith("Your hand:"):
                    chat_lines.append("You have been dealt your private cards.")
        except:
            break


# === Connect to server ===
ip, port = discover_server()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ip, port))

threading.Thread(target=receive_messages, args=(sock,), daemon=True).start()

# === Ask for name via Pygame input ===
chat_lines.append("Enter your name and press Enter:")
entering_name = True

# === Main loop ===
running = True
while running:
    draw_ui()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if entering_name:
                    sock.send(input_text.encode())
                    chat_lines.append(f"Name sent: {input_text}")
                    entering_name = False
                    input_text = ''
                else:
                    pass  # no input used in lobby phase
            elif event.key == pygame.K_BACKSPACE:
                input_text = input_text[:-1]
            else:
                if len(input_text) < 50:
                    input_text += event.unicode

    clock.tick(30)

sock.close()
pygame.quit()
