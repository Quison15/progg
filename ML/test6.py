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
SPEED = 20  # Kan justeras för att göra spelet lite långsammare under träning
WIDTH, HEIGHT = 200, 200

# Grid-dimensioner
GRID_WIDTH = WIDTH // BLOCK_SIZE   # ex. 20
GRID_HEIGHT = HEIGHT // BLOCK_SIZE   # ex. 20

# Max antal kroppsegment (exklusive huvudet)
MAX_BODY = 10

# Riktningar (numpy-arrayer)
DIRECTION_LEFT  = np.array([-1, 0])
DIRECTION_RIGHT = np.array([1, 0])
DIRECTION_UP    = np.array([0, -1])
DIRECTION_DOWN  = np.array([0, 1])

# För enkel hantering av riktningar
directions = {
    "RIGHT": DIRECTION_RIGHT,
    "LEFT":  DIRECTION_LEFT,
    "UP":    DIRECTION_UP,
    "DOWN":  DIRECTION_DOWN,
}

###########################
# SPELETS MILJÖ (ENVIRONMENT)
###########################

class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI - FC Q-Learning (Full Positionsinfo)")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = directions["RIGHT"]
        # Placera huvudet i mitten
        self.head = np.array([self.w // (2 * BLOCK_SIZE), self.h // (2 * BLOCK_SIZE)])
        self.snake = [self.head.copy()]  # Första elementet är huvudet
        self.spawn_food()
        self.score = 0
        self.frame_iteration = 0
        self.game_over = False

    def spawn_food(self):
        while True:
            x = random.randint(0, (self.w // BLOCK_SIZE) - 1)
            y = random.randint(0, (self.h // BLOCK_SIZE) - 1)
            self.food = np.array([x, y])
            # Se till att maten inte spawnar på ormen
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
        Utför ett steg i spelet.
        action: one-hot vektor [1,0,0]=rakt, [0,1,0]=sväng höger, [0,0,1]=sväng vänster.
        Returnerar (reward, game_over, score)
        """
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self._move(action)
        self.snake.insert(0, self.head.copy())

        if self.is_collision():
            self.game_over = True
            reward = -10
            return reward, self.game_over, self.score

        if (self.head == self.food).all():
            self.score += 1
            reward = 100
            self.spawn_food()
        else:
            reward = 0
            self.snake.pop()

        self.update_ui()
        self.clock.tick(SPEED)

        # Timeout för att undvika oändliga loopar
        if self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -10

        return reward, self.game_over, self.score

    def _move(self, action):
        """
        Uppdaterar riktning baserat på handlingen.
        """
        # Riktningarna i klockvis ordning: höger, ned, vänster, upp
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

###########################################
# STATE-REPRESENTATION MED FULLA POSITIONER
###########################################

def get_state_full(game):
    """
    Returnerar en state-vektor med följande information (alla värden normaliserade till [0,1]):
      - Huvudets position: (x, y)
      - Äpplets position: (x, y)
      - Väggarnas avstånd från huvudet: (vänster, höger, upp, ned)
      - Kroppens positioner (upp till MAX_BODY segment, varje segment: (x, y); om färre än MAX_BODY, pad med 0)
    Total dimension: 2 + 2 + 4 + (MAX_BODY * 2) = 108.
    """
    # Normalisera genom att dividera med grid-dimension
    head = game.head.astype(np.float32) / np.array([GRID_WIDTH, GRID_HEIGHT])
    apple = game.food.astype(np.float32) / np.array([GRID_WIDTH, GRID_HEIGHT])
    
    # Väggarnas avstånd från huvudet (i antal celler, normaliserade)
    # Antag att väggarna ligger vid index 0 och GRID_WIDTH-1 / GRID_HEIGHT-1
    left_dist   = game.head[0] / (GRID_WIDTH - 1)
    right_dist  = (GRID_WIDTH - 1 - game.head[0]) / (GRID_WIDTH - 1)
    top_dist    = game.head[1] / (GRID_HEIGHT - 1)
    bottom_dist = (GRID_HEIGHT - 1 - game.head[1]) / (GRID_HEIGHT - 1)
    walls = np.array([left_dist, right_dist, top_dist, bottom_dist], dtype=np.float32)
    
    # Kroppens positioner (exklusive huvudet)
    body = []
    for segment in game.snake[1:]:
        seg = segment.astype(np.float32) / np.array([GRID_WIDTH, GRID_HEIGHT])
        body.extend(seg.tolist())
    # Padding om längden är mindre än MAX_BODY
    # Vi förväntar oss MAX_BODY*2 värden
    if len(body) < MAX_BODY * 2:
        body.extend([0.0] * (MAX_BODY * 2 - len(body)))
    else:
        body = body[:MAX_BODY * 2]
    body = np.array(body, dtype=np.float32)
    
    state = np.concatenate((head, apple, walls, body))
    return state  # Form: (108,)

###########################################
# FULLY CONNECTED NEURALT NÄTVERK (FC)
###########################################

class FC_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FC_QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save(self, file_name='fc_model.pth'):
        torch.save(self.state_dict(), file_name)

###########################################
# Q-LEARNING TRAINER
###########################################

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        # Om state är en enskild sample, lägg till en batch-dimension
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
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

###########################################
# AGENT MED FULLA POSITIONER
###########################################

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Utforskningsfaktor (epsilon-greedy)
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=100_000)
        self.batch_size = 64
        self.model = FC_QNet(input_size=28, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
    
    def get_state(self, game):
        state = get_state_full(game)
        return torch.tensor(state, dtype=torch.float)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
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
        # Epsilon-greedy strategi: högre utforskning i början
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = state.unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

###########################################
# TRÄNINGSLOOP
###########################################

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
