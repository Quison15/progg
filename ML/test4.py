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

# Globala konstanter
BLOCK_SIZE = 20
SPEED = 40  # Spelhastighet
WIDTH, HEIGHT = 400, 400

# Beräkna grid-dimensioner
GRID_WIDTH = WIDTH // BLOCK_SIZE   # ex. 20
GRID_HEIGHT = HEIGHT // BLOCK_SIZE   # ex. 20

# Definition av cellkoder:
# 0: Tom cell, 1: Vägg, 2: Ormdel, 3: Frukt

# Riktningar som numpy-arrayer
DIRECTION_LEFT  = np.array([-1, 0])
DIRECTION_RIGHT = np.array([1, 0])
DIRECTION_UP    = np.array([0, -1])
DIRECTION_DOWN  = np.array([0, 1])

###########################
# SPELETS MILJÖ (ENVIRONMENT)
###########################

class SnakeGameAI:
    """
    Spelmiljö för Snake med grid-representation.
    """
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI - CNN & Deep Q-Learning")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = DIRECTION_RIGHT
        self.head = np.array([self.w // (2 * BLOCK_SIZE), self.h // (2 * BLOCK_SIZE)])  # Starta i mitten
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
        self.display.fill((0, 0, 0))
        # Rita ormen
        for s in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), (s[0]*BLOCK_SIZE, s[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Rita maten
        pygame.draw.rect(self.display, (255, 0, 0), (self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        font = pygame.font.SysFont('arial', 25)
        text = font.render("Score: " + str(self.score), True, (255,255,255))
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
        Utför ett steg i spelet baserat på agentens handling.
        action: En one-hot vektor: [1,0,0] = fortsätt rakt, [0,1,0] = sväng höger, [0,0,1] = sväng vänster.
        Returnerar (reward, game_over, score)
        """
        self.frame_iteration += 1

        # Hantera Pygame-händelser
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Utför rörelsen
        self._move(action)
        self.snake.insert(0, self.head.copy())

        # Kolla kollision
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
            reward = 0  # Ingen extra reward om inget händer
            self.snake.pop()

        self.update_ui()
        self.clock.tick(SPEED)

        # Timeout om spelet drar ut på tiden
        if self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -10

        return reward, self.game_over, self.score

    def _move(self, action):
        """
        Uppdaterar self.direction och self.head baserat på handlingen.
        action: [1,0,0] = rakt, [0,1,0] = höger, [0,0,1] = vänster.
        """
        # Ordning: höger, ned, vänster, upp (klockvis)
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
    Returnerar spelplanens grid som en 2D-array med dimension (GRID_HEIGHT, GRID_WIDTH)
    med koder:
      0: Tom cell
      1: Vägg (border)
      2: Ormdel
      3: Frukt
    """
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    # Sätt väggar (border)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Placera ormen (sätt värde 2)
    for s in game.snake:
        x, y = s
        if 0 < x < GRID_WIDTH-1 and 0 < y < GRID_HEIGHT-1:
            grid[y, x] = 2

    # Placera maten (sätt värde 3)
    fruit_x, fruit_y = game.food
    if 0 < fruit_x < GRID_WIDTH-1 and 0 < fruit_y < GRID_HEIGHT-1:
        grid[fruit_y, fruit_x] = 3

    return grid

def one_hot_state(grid):
    """
    One-hot-encodar gridet. Antalet kanaler = 4.
    Returnerar en tensor med shape (4, GRID_HEIGHT, GRID_WIDTH)
    """
    grid_tensor = torch.tensor(grid, dtype=torch.long)  # shape: (H, W)
    one_hot = F.one_hot(grid_tensor, num_classes=4)      # shape: (H, W, 4)
    one_hot = one_hot.permute(2, 0, 1).float()             # shape: (4, H, W)
    return one_hot

###########################
# CNN-BASERAT NEURALT NÄTVERK
###########################

class CNN_QNet(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN_QNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # Output: 32 x H x W
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)              # Output: 64 x H x W
        self.fc_input_dim = 64 * GRID_HEIGHT * GRID_WIDTH
        self.fc = nn.Linear(self.fc_input_dim, output_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))         # (batch, 32, H, W)
        x = F.relu(self.conv2(x))         # (batch, 64, H, W)
        x = x.view(x.size(0), -1)         # Flat: (batch, 64*H*W)
        x = self.fc(x)                    # (batch, output_size)
        return x
    
    def save(self, file_name='cnn_model.pth'):
        torch.save(self.state_dict(), file_name)

###########################
# Q-LEARNING TRAINER
###########################

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        # Om state är en enskild sample (d.v.s. shape == (channels, H, W)), lägg till batch-dimension
        if len(state.shape) == 3:
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
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

###########################
# AGENT MED CNN
###########################

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Utforskningsfaktor
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=100_000)
        self.batch_size = 64
        # Använder CNN_QNet med 4 ingångskanaler (one-hot för grid) och 3 utgångar
        self.model = CNN_QNet(input_channels=4, output_size=3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
    
    def get_state(self, game):
        grid = get_state_grid(game)          # 2D-array med dimension (GRID_HEIGHT, GRID_WIDTH)
        state = one_hot_state(grid)          # Tensor med shape (4, GRID_HEIGHT, GRID_WIDTH)
        return state
    
    def remember(self, state, action, reward, next_state, done):
        # För att spara minnet konverterar vi state till numpy om nödvändigt
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # Konvertera listor av tensors till en batch-tensor
        states = torch.stack(states)
        actions = torch.tensor(np.array(actions), dtype=torch.float)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.stack(next_states)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, 
                                torch.tensor(action, dtype=torch.float),
                                torch.tensor(reward, dtype=torch.float),
                                next_state,
                                done)
    
    def get_action(self, state):
        # Justera epsilon baserat på antalet spel
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Lägg till batch-dimension
            state0 = state.unsqueeze(0)
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
