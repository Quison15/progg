from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import math


class Agent:
    def __init__(self,env: gym.Env,learning_rate: float,initial_epsilon: float,epsilon_decay: float,final_epsilon: float,discount_factor: float = 0.95):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    def get_action(self, obs: np.ndarray) -> int:
        state = self.discretize(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[state]))
    def save(self):
        np.save("q_values.npy",self.q_values)
        
    def update(self, obs: np.ndarray,action: int, reward: float, terminated: bool, next_obs: np.ndarray):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_state])

        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.q_values[state][action]

        self.q_values[state][action] = (self.q_values[state][action] + self.lr * temporal_difference)
        
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def discretize(self, obs: np.ndarray, bins=(6, 12, 6, 12)) -> tuple:
        """Convert continuous state into a discrete one."""
        upper_bounds = [self.env.observation_space.high[0], 0.5,
                        self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5,
                        self.env.observation_space.low[2], -math.radians(50)]
        
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

# Training hyperparameters
learning_rate = 0.5        # How fast to learn (higher = faster but less stable)
n_episodes = 50000       # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration


env = gym.make('CartPole-v1', render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = Agent(env=env,learning_rate=learning_rate,initial_epsilon=start_epsilon,epsilon_decay=epsilon_decay,final_epsilon=final_epsilon)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs,action,reward,terminated,next_obs)

        done = terminated or truncated
        obs = next_obs
    
    agent.decay_epsilon()
env.close()

env_test = gym.make("CartPole-v1", render_mode="human")

def test_agent(agent: Agent,episodes=100):
    for episode in range(episodes):
        obs = env_test.reset()[0]  # Reset the environment
        state1 = agent.discretize(obs)
        done = False
        steps = 0

        while not done:  # Visualize the agent's performance
            action = np.argmax(agent.q_values[state1])  # Use the trained policy
            state, _, done, _, _ = env_test.step(action)  # Take the best action (based on learned Q-values)
            steps += 1
        
        print(f"Episode {episode + 1}: Agent balanced for {steps} steps.")
    
    env_test.close()

# Test the agent
test_agent(agent,episodes=100)


