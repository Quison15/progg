import pygame
import random
import numpy as np
from collections import defaultdict
import sys

# Initiera Pygame
pygame.init()

# Globala konstanter
BLOCK_SIZE = 20
SPEED = 40  # Spelhastighet
WIDTH, HEIGHT = 400, 400

# Färger
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Riktningar som numpy-arrayer
DIRECTION_LEFT = np.array([-1, 0])
DIRECTION_RIGHT = np.array([1, 0])
DIRECTION_UP = np.array([0, -1])
DIRECTION_DOWN = np.array([0, 1])

class SnakeGameAI:
    """
    En klass för att hantera spelets logik (Snake) i en AI-miljö.
    """
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI - Q-learning")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = DIRECTION_RIGHT
        self.head = np.array([self.w // (2 * BLOCK_SIZE), self.h // (2 * BLOCK_SIZE)])  # Start i mitten
        self.snake = [self.head.copy()]
        self.spawn_food()
        self.score = 0
        self.frame_iteration = 0
        self.game_over = False

    def spawn_food(self):
        while True:
            x = random.randint(0, (self.w // BLOCK_SIZE) - 1)
            y = random.randint(0, (self.h // BLOCK_SIZE) - 1)
            self.food = np.array([x, y])
            if not any((self.food == s).all() for s in self.snake):
                break

    def update_ui(self):
        self.display.fill(BLACK)
        for s in self.snake:
            pygame.draw.rect(self.display, GREEN, (s[0] * BLOCK_SIZE, s[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, (self.food[0] * BLOCK_SIZE, self.food[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Visa poäng
        font = pygame.font.SysFont('arial', 25)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Kollision med väggar
        if pt[0] < 0 or pt[0] >= self.w // BLOCK_SIZE or pt[1] < 0 or pt[1] >= self.h // BLOCK_SIZE:
            return True
        # Kollision med sig själv
        if any((pt == s).all() for s in self.snake[1:]):
            return True
        return False

    def play_step(self, action):
        """
        Utför ett steg i spelet baserat på den givna åtgärden.
        action: En one-hot vektor [1,0,0]=rakt, [0,1,0]=sväng höger, [0,0,1]=sväng vänster.
        Returnerar (reward, game_over, score).
        """
        self.frame_iteration += 1

        # Hantera Pygame-händelser
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Uppdatera riktning och huvudposition utifrån åtgärden
        self._move(action)
        self.snake.insert(0, self.head.copy())

        reward = 0
        # Kontrollera kollision (game over)
        if self.is_collision():
            self.game_over = True
            reward = -10
            return reward, self.game_over, self.score

        # Om ormen äter maten
        if (self.head == self.food).all():
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()

        self.update_ui()
        self.clock.tick(SPEED)

        # Förhindra att spelet drar ut på tiden
        if self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -10

        return reward, self.game_over, self.score

    def _move(self, action):
        """
        Uppdatera self.direction och self.head baserat på agentens val.
        action: [1,0,0] = fortsätt rakt, [0,1,0] = sväng höger, [0,0,1] = sväng vänster.
        """
        # Ordning: höger, ned, vänster, upp (klockvis)
        clock_wise = [DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_UP]
        # Konvertera self.direction och elementen i clock_wise med .tolist() för entydiga jämförelser
        direction_tuples = [tuple(d.tolist()) for d in clock_wise]
        current_direction = tuple(self.direction.tolist())
        try:
            idx = direction_tuples.index(current_direction)
        except ValueError:
            idx = 0

        if np.array_equal(action, [1, 0, 0]):
            # Ingen ändring
            new_dir = self.direction
        elif np.array_equal(action, [0, 1, 0]):
            # Sväng höger
            new_dir = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):
            # Sväng vänster
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = self.direction

        self.direction = new_dir
        self.head = self.head + self.direction

def get_state(game):
    """
    Returnera en tuple som representerar spelets tillstånd.
    Inkluderar:
    - Fara rakt fram, höger, vänster
    - Nuvarande riktning
    - Matens position relativt huvudet
    """
    head = game.head

    # Direkt kontroller för faror
    danger_straight = game.is_collision(head + game.direction)

    # Bestäm faror beroende på riktning
    if np.array_equal(game.direction, DIRECTION_UP):
        danger_right = game.is_collision(head + DIRECTION_RIGHT)
        danger_left = game.is_collision(head + DIRECTION_LEFT)
    elif np.array_equal(game.direction, DIRECTION_DOWN):
        danger_right = game.is_collision(head + DIRECTION_LEFT)
        danger_left = game.is_collision(head + DIRECTION_RIGHT)
    elif np.array_equal(game.direction, DIRECTION_LEFT):
        danger_right = game.is_collision(head + DIRECTION_UP)
        danger_left = game.is_collision(head + DIRECTION_DOWN)
    elif np.array_equal(game.direction, DIRECTION_RIGHT):
        danger_right = game.is_collision(head + DIRECTION_DOWN)
        danger_left = game.is_collision(head + DIRECTION_UP)
    else:
        danger_right = False
        danger_left = False

    # Riktningar
    dir_left = np.array_equal(game.direction, DIRECTION_LEFT)
    dir_right = np.array_equal(game.direction, DIRECTION_RIGHT)
    dir_up = np.array_equal(game.direction, DIRECTION_UP)
    dir_down = np.array_equal(game.direction, DIRECTION_DOWN)

    # Matens position relativt ormens huvud
    food_left = game.food[0] < head[0]
    food_right = game.food[0] > head[0]
    food_up = game.food[1] < head[1]
    food_down = game.food[1] > head[1]

    state = (
        danger_straight,
        danger_right,
        danger_left,
        dir_left,
        dir_right,
        dir_up,
        dir_down,
        food_left,
        food_right,
        food_up,
        food_down
    )
    return state

class Agent:
    """
    Agenten implementerar Q-learning med en Q-tabell (lagrad som en dictionary).
    """
    def __init__(self, learning_rate=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = defaultdict(lambda: np.zeros(3))  # Tre möjliga handlingar
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_action(self, state):
        state = tuple(state)
        # Epsilon-girig strategi: välj slumpmässigt med sannolikhet epsilon
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 2)
        else:
            action_idx = np.argmax(self.q_table[state])
        # Returnera one-hot vektor
        action = [0, 0, 0]
        action[action_idx] = 1
        return np.array(action)

    def update_q_value(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)
        action_idx = np.argmax(action)
        q_predict = self.q_table[state][action_idx]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_idx] += self.learning_rate * (q_target - q_predict)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def train():
    episodes = 1000  # Antal träningsomgångar
    agent = Agent()
    game = SnakeGameAI()
    scores = []

    for episode in range(episodes):
        game.reset()
        state_old = get_state(game)
        total_reward = 0

        while True:
            # Hämta åtgärd från agenten
            action = agent.get_action(state_old)
            # Utför ett steg i spelet med vald åtgärd
            reward, done, score = game.play_step(action)
            state_new = get_state(game)
            # Uppdatera Q-tabellen med Q-learning-regeln
            agent.update_q_value(state_old, action, reward, state_new, done)
            state_old = state_new
            total_reward += reward
            if done:
                break
        
        agent.decay_epsilon()
        scores.append(score)
        print(f"Episode: {episode + 1}, Score: {score}, Epsilon: {agent.epsilon:.4f}")
    
    pygame.quit()

if __name__ == '__main__':
    train()
