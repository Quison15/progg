import pygame
import random
import numpy as np
import sys
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Initiera Pygame
pygame.init()

# Globala konstanter för spelet
BLOCK_SIZE = 20
SPEED = 40  # Spelhastighet
WIDTH, HEIGHT = 200, 200

# Riktningar som numpy-arrayer
DIRECTION_LEFT = np.array([-1, 0])
DIRECTION_RIGHT = np.array([1, 0])
DIRECTION_UP = np.array([0, -1])
DIRECTION_DOWN = np.array([0, 1])

# Färger
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Beräkna griddimensioner
GRID_WIDTH = WIDTH // BLOCK_SIZE   # t.ex. 20
GRID_HEIGHT = HEIGHT // BLOCK_SIZE   # t.ex. 20

###########################
# SPELETS MILJÖ (ENVIRONMENT)
###########################

class SnakeGameAI:
    """
    Hanterar spelets logik för Snake med AI.
    """
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI - Deep Q-Learning (Grid State)")
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
            pygame.draw.rect(self.display, GREEN, (s[0]*BLOCK_SIZE, s[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, (self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        font = pygame.font.SysFont('arial', 25)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        grid_w = self.w // BLOCK_SIZE
        grid_h = self.h // BLOCK_SIZE
        if pt[0] < 0 or pt[0] >= grid_w or pt[1] < 0 or pt[1] >= grid_h:
            return True
        if any((pt == s).all() for s in self.snake[1:]):
            return True
        return False

    def play_step(self, action):
        """
        Utför ett steg i spelet med den givna åtgärden.
        action: En one-hot vektor [1,0,0]=rakt, [0,1,0]=sväng höger, [0,0,1]=sväng vänster.
        
        Belöningsformulering:
         - Vi mäter avståndet från huvudet till maten före och efter rörelsen.
         - Om ormen närmar sig maten ges +0.1, annars -0.1.
         - Vid kollision ges -10 och om maten äts ges +10.
        """
        self.frame_iteration += 1

        # Hantera Pygame-händelser
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Mät avståndet innan rörelsen
        prev_distance = np.linalg.norm(self.head - self.food)
        self._move(action)
        self.snake.insert(0, self.head.copy())
        new_distance = np.linalg.norm(self.head - self.food)
        
        if self.is_collision():
            self.game_over = True
            reward = -10
            return reward, self.game_over, self.score

        if (self.head == self.food).all():
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            shaping_reward = 0.1 if new_distance < prev_distance else -0.1
            reward = shaping_reward
            self.snake.pop()

        self.update_ui()
        self.clock.tick(SPEED)

        if self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -10

        return reward, self.game_over, self.score

    def _move(self, action):
        """
        Uppdaterar self.direction och self.head baserat på agentens val.
        action: [1,0,0] = fortsätt rakt, [0,1,0] = sväng höger, [0,0,1] = sväng vänster.
        """
        clock_wise = [DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_UP]
        direction_tuples = [tuple(d.tolist()) for d in clock_wise]
        current_direction = tuple(self.direction.tolist())
        try:
            idx = direction_tuples.index(current_direction)
        except ValueError:
            idx = 0

        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.direction
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = self.direction

        self.direction = new_dir
        self.head = self.head + self.direction

def get_state_grid(game):
    """
    Returnerar spelplanens fullständiga grid som en 1D-array.
    Kodning:
      0: Tom cell
      1: Vägg (border)
      2: Ormdel
      3: Mat (frukt)
    """
    grid_w = game.w // BLOCK_SIZE
    grid_h = game.h // BLOCK_SIZE
    state_grid = np.zeros((grid_h, grid_w), dtype=int)
    
    # Sätt väggar (border)
    state_grid[0, :] = 1
    state_grid[-1, :] = 1
    state_grid[:, 0] = 1
    state_grid[:, -1] = 1

    # Placera ormen
    for s in game.snake:
        x, y = s
        if 0 < y < grid_h - 1 and 0 < x < grid_w - 1:
            state_grid[y, x] = 2

    # Placera maten
    fruit_x, fruit_y = game.food
    if 0 < fruit_y < grid_h - 1 and 0 < fruit_x < grid_w - 1:
        state_grid[fruit_y, fruit_x] = 3

    return state_grid.flatten()

###########################
# DEEP Q-LEARNING DELEN
###########################

class Linear_QNet(nn.Module):
    """
    Enkel fullt ansluten nätverksmodell.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

class QTrainer:
    """
    Träningsklass för att optimera Q-nätverket.
    """
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)
        
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class Agent:
    """
    Agenten med replay memory och Deep Q-Learning.
    """
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Utforskningsfaktor (epsilon-greedy)
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=100_000)  # Replay memory
        self.batch_size = 64
        # Inputstorleken blir nu GRID_WIDTH * GRID_HEIGHT (t.ex. 20x20 = 400)
        input_size = GRID_WIDTH * GRID_HEIGHT
        self.model = Linear_QNet(input_size=input_size, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
    
    def get_state(self, game):
        return get_state_grid(game)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.float)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

###########################
# TRÄNINGSLOOP
###########################

def train():
    agent = Agent()
    game = SnakeGameAI()
    total_score = 0
    record = 0
    episodes = 1000

    for episode in range(episodes):
        game.reset()
        state_old = agent.get_state(game)
        done = False

        while not done:
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(np.array(action))
            state_new = agent.get_state(game)
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            state_old = state_new
        
        agent.train_long_memory()
        
        if score > record:
            record = score
            agent.model.save()
        
        agent.n_games += 1
        total_score += score
        mean_score = total_score / agent.n_games
        print(f"Episode: {agent.n_games}, Score: {score}, Record: {record}, Mean Score: {mean_score:.2f}")
    
    pygame.quit()

if __name__ == '__main__':
    train()
