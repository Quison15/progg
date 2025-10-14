import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def run(episodes, is_training, render):

    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, success_rate=1.0/3.0, reward_schedule=(1, 0, 0), render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('sarsa_frozen_lake.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1
    final_epsilon = 0

    #epsilon_decay_rate = epsilon / (episodes / 1.5)
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    
    def take_action(state):
        if is_training and rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state,:])
        return action

    for i in tqdm(range(episodes)):

        state = env.reset()[0]
        action = take_action(state)
        done = False

        rewards = 0
        while not done:
            

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            next_action = take_action(next_state)
            if is_training:
                q[state,action] += + learning_rate * (
                    reward + discount_factor*q[next_state, next_action] - q[state,action]
                )
            state = next_state
            action = next_action
            rewards += reward
        epsilon = max(final_epsilon, epsilon - epsilon_decay_rate)
        rewards_per_episode[i] = rewards
    env.close()
    

    if is_training:
        f = open('sarsa_frozen_lake.pkl', 'wb')
        pickle.dump(q,f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'sarsa_frozen_lake.png')



#run(15000, is_training=True, render=False)
run(5, is_training=False, render=True)