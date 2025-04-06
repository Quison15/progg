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
SPEED = 400
WIDTH, HEIGHT = 400, 400

# Riktningar (som numpy-arrayer)
DIRECTION_LEFT  = np.array([-1, 0])
DIRECTION_RIGHT = np.array([1, 0])
DIRECTION_UP    = np.array([0, -1])
DIRECTION_DOWN  = np.array([0, 1])

# Hjälp-dictionary för riktningar
directions = {
    "RIGHT": DIRECTION_RIGHT,
    "LEFT":  DIRECTION_LEFT,
    "UP":    DIRECTION_UP,
    "DOWN":  DIRECTION_DOWN,
}


##########################################
# BFS för att beräkna fritt utrymme från huvudet
##########################################
def calculate_free_space(game):
    """
    Använder en enkel BFS (breadth-first search) för att räkna antalet tillgängliga celler från ormens huvud,
    utan att korsa väggar eller ormens kropp.
    """
    grid_w = game.w // BLOCK_SIZE
    grid_h = game.h // BLOCK_SIZE
    visited = set()
    queue = [tuple(game.head)]
    free = 0

    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        # Om positionen ligger utanför gränserna, hoppa över
        if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
            continue
        # Om cellen är upptagen av ormen, hoppa över
        if any((np.array([x, y]) == part).all() for part in game.snake):
            continue
        free += 1
        # Lägg till grannar (upp, ned, vänster, höger)
        queue.append((x+1, y))
        queue.append((x-1, y))
        queue.append((x, y+1))
        queue.append((x, y-1))
    return free


##########################################
# SPELETS MILJÖ (ENVIRONMENT)
##########################################
class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI med BFS-belöning")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = directions["RIGHT"]
        # Placera huvudet i mitten
        self.head = np.array([self.w // (2 * BLOCK_SIZE), self.h // (2 * BLOCK_SIZE)])
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
            # Se till att maten inte placeras på ormen
            if not any((self.food == part).all() for part in self.snake):
                break

    def update_ui(self):
        self.display.fill((0, 0, 0))
        for part in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), (part[0]*BLOCK_SIZE, part[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (255, 0, 0), (self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        font = pygame.font.SysFont('arial', 25)
        text = font.render("Score: " + str(self.score), True, (255,255,255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Kollision med väggarna
        if pt[0] < 0 or pt[0] >= self.w // BLOCK_SIZE or pt[1] < 0 or pt[1] >= self.h // BLOCK_SIZE:
            return True
        # Kollision med ormen själv
        if any((pt == part).all() for part in self.snake[1:]):
            return True
        return False

    def play_step(self, action):
        """
        Utför ett steg i spelet med den givna handlingen.
        action: one-hot vektor [1,0,0] = rakt, [0,1,0] = sväng höger, [0,0,1] = sväng vänster.
        Inkluderar en belöningsfunktion som kombinerar:
          - Klassisk belöning: +10 om maten äts, -10 vid kollision/timeout.
          - En BFS-baserad belöning: om fritt utrymme från huvudet ökar ges en bonus,
            om det minskar ges ett straff.
          - En liten överlevnadsbonus per steg.
        Returnerar (reward, game_over, score)
        """
        self.frame_iteration += 1

        # Hantera händelser
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Mät fritt utrymme före rörelsen
        free_space_before = calculate_free_space(self)

        # Utför rörelsen
        self._move(action)
        self.snake.insert(0, self.head.copy())

        # Om kollision inträffar
        if self.is_collision():
            self.game_over = True
            reward = -10
            return reward, self.game_over, self.score

        # Om maten äts
        if (self.head == self.food).all():
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()
            # Mät fritt utrymme efter rörelsen
            free_space_after = calculate_free_space(self)
            # Beräkna belöningen baserat på förändringen i fritt utrymme
            space_reward = 0.01 * (free_space_after - free_space_before)
            # Överlevnadsbonus per steg
            survival_bonus = 0.05
            #reward = space_reward + survival_bonus
            reward = space_reward + survival_bonus
            print(free_space_before, free_space_after)

        self.update_ui()
        self.clock.tick(SPEED)

        # Timeout för att förhindra oändliga loopar
        if self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -10

        return reward, self.game_over, self.score

    def _move(self, action):
        """
        Uppdaterar self.direction och self.head baserat på handlingen.
        """
        # Riktningar i klockvis ordning: höger, ned, vänster, upp
        clock_wise = [DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_UP]
        direction_tuples = [tuple(d.tolist()) for d in clock_wise]
        current_direction = tuple(self.direction.tolist())
        try:
            idx = direction_tuples.index(current_direction)
        except ValueError:
            idx = 0

        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.direction  # Ingen förändring
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # Sväng höger
        elif np.array_equal(action, [0, 0, 1]):
            new_dir = clock_wise[(idx - 1) % 4]  # Sväng vänster
        else:
            new_dir = self.direction

        self.direction = new_dir
        self.head = self.head + self.direction


##########################################
# STATE-REPRESENTATION (11 dimensioner)
##########################################
def get_state(game):
    """
    Returnerar en 11-dimensionell state-vektor med följande information:
      - Fara: om det är fara rakt fram, åt höger eller åt vänster (booleska värden)
      - Riktning: om ormen rör sig vänster, höger, upp eller ner (booleska värden)
      - Matens position relativt huvudet: (vänster, höger, upp, ned) (booleska värden)
    """
    head = game.head

    # Fara direkt (om ormen går rakt, åt höger eller åt vänster och krockar)
    danger_straight = game.is_collision(head + game.direction)

    if np.array_equal(game.direction, DIRECTION_UP):
        danger_right = game.is_collision(head + DIRECTION_RIGHT)
        danger_left  = game.is_collision(head + DIRECTION_LEFT)
    elif np.array_equal(game.direction, DIRECTION_DOWN):
        danger_right = game.is_collision(head + DIRECTION_LEFT)
        danger_left  = game.is_collision(head + DIRECTION_RIGHT)
    elif np.array_equal(game.direction, DIRECTION_LEFT):
        danger_right = game.is_collision(head + DIRECTION_UP)
        danger_left  = game.is_collision(head + DIRECTION_DOWN)
    elif np.array_equal(game.direction, DIRECTION_RIGHT):
        danger_right = game.is_collision(head + DIRECTION_DOWN)
        danger_left  = game.is_collision(head + DIRECTION_UP)
    else:
        danger_right = False
        danger_left  = False

    # Riktning
    dir_left  = np.array_equal(game.direction, DIRECTION_LEFT)
    dir_right = np.array_equal(game.direction, DIRECTION_RIGHT)
    dir_up    = np.array_equal(game.direction, DIRECTION_UP)
    dir_down  = np.array_equal(game.direction, DIRECTION_DOWN)

    # Matens position relativt huvudet
    food_left  = game.food[0] < head[0]
    food_right = game.food[0] > head[0]
    food_up    = game.food[1] < head[1]
    food_down  = game.food[1] > head[1]

    state = [
        int(danger_straight),
        int(danger_right),
        int(danger_left),
        int(dir_left),
        int(dir_right),
        int(dir_up),
        int(dir_down),
        int(food_left),
        int(food_right),
        int(food_up),
        int(food_down)
    ]
    return np.array(state, dtype=int)


##########################################
# NEURALT NÄTVERK & Q-LEARNING
##########################################
class Linear_QNet(nn.Module):
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
    def __init__(self, model, lr, gamma):
        self.model  = model
        self.lr     = lr
        self.gamma  = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        if len(state.shape) == 1:
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = (done,)
        
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
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Utforskningsfaktor
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=100_000)
        self.batch_size = 64
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
    
    def get_state(self, game):
        return torch.tensor(get_state(game), dtype=torch.float)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states      = torch.stack(states)
        actions     = torch.tensor(np.array(actions), dtype=torch.float)
        rewards     = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.stack(next_states)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, 
                                torch.tensor(action, dtype=torch.float),
                                torch.tensor(reward, dtype=torch.float),
                                next_state,
                                done)
    
    def get_action(self, state):
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


##########################################
# TRÄNINGSLOOP
##########################################
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
