import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def run(episodes, is_training = True, render=False):

    env = gym.make("MountainCar-v0", render_mode='human' if render else None)
    #env = gym.wrappers.TransformReward(env)

    #pos_bins = 50
    #vel_bins = 50
    #win_size = (env.observation_space.high - env.observation_space.low) / [pos_bins, vel_bins]
    #def get_discrete_state(state):
    #    return tuple(((state - env.observation_space.low) / win_size).astype(int))

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0],125)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1],125)

    if is_training:
        #q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) #20x20x3 array
        q = np.random.uniform(low=-2, high=0, size=(len(pos_space), len(vel_space), env.action_space.n))
    else:
        f = open('sarsa_mountain_car.pkl', 'rb')
        q = pickle.load(f)
        f.close()
        #q = np.load('best_qtable.npy', allow_pickle=True)
    learning_rate = 0.1 #alpha
    discount_factor = 0.999 #gamma

    epsilon = 0.0 #100% random actions
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    
    def take_action(state_p,state_v):
        if is_training and rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state_p,state_v,:])
        return action

    for i in tqdm(range(episodes)):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False

        rewards = 0
        action = take_action(state_p,state_v)
        while(not terminated and rewards>-200):
            
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            next_action = take_action(new_state_p,new_state_v)
            if is_training:
                q[state_p,state_v,action] +=  + learning_rate * (
                    reward + discount_factor*np.max(q[new_state_p,new_state_v,next_action]) - q[state_p,state_v,action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            action = next_action

            rewards += reward
        
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

    env.close()

    if is_training:
        f = open('sarsa_mountain_car.pkl', 'wb')
        pickle.dump(q,f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'sarsa_mountain_car.png')

#run(250000, is_training=True, render=False)
run(10, is_training=False, render=True)