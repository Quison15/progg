import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


def run(episodes, is_training, render):

    env = gym.make('Taxi-v3',is_rainy=True,fickle_passenger=True, render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('taxi.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    learning_rate = 0.9
    discount_factor = 0.95
    epsilon = 1
    final_epsilon = 0

    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in tqdm(range(episodes)):
        state = env.reset()[0]

        done = False

        rewards = 0
        while not done:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])
            next_state, reward, terminated, truncated, info = env.step(action)
            if is_training:
                q[state,action] = q[state,action] + learning_rate * (
                    reward + discount_factor*np.max(q[next_state,:]) - q[state,action]
                )
            done = terminated or truncated
            state = next_state

            rewards += reward
        epsilon = max(final_epsilon, epsilon - epsilon_decay_rate)
        rewards_per_episode[i] = rewards
    env.close()

    if is_training:
        f = open('taxi.pkl', 'wb')
        pickle.dump(q,f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'taxi.png')


#run(15000, is_training=True, render=False)
run(10, is_training=False, render=True)