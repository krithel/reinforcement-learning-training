# Tiling
import random

def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    assert len(low) == len(high) == len(bins) == len(offsets)
    
    dims = len(low)
    
    split_points = []
    for i in range(dims):
        step = (high[i] - low[i])/bins[i]
        
        split_points.append(np.linspace(low[i]+offsets[i]+step, high[i]+offsets[i], bins[i]-1, False))
                            
    return np.array(split_points)

def tile_encode(sample, high, low, tiling_specs, flatten=False):
    tilings = np.array([create_tiling_grid(low, high, tiling_specs[i][0], tiling_specs[i][1]) for i in range(len(tiling_specs))])
    encoded_sample = [[int(np.digitize(s, g)) for s, g in zip(sample, grid)] for grid in tilings]
    features = []
    for s, spec in zip(encoded_sample, tiling_specs):
        f = np.zeros(spec[0])
        f[tuple(s)] = 1
        features.append(f.flatten())
        
#     print(features)
    return np.concatenate(features)
#     one_hot = [[1 if i == index else 0 for index, bins in zip(sample, spec[0]) for i in range(bins)] for sample, spec in zip(encoded_sample, tiling_specs)]    
#     return np.concatenate(one_hot) if flatten else one_hot

def action_value_delta(state, action, weights, high, low, tiling_specs):
    state_action = (state[0], state[1], action)
    return tile_encode(state_action, high, low, tiling_specs, True)

def action_value_approx(state, action, weights, high, low, tiling_specs):
    # State = (float, float)
    # Action = 0 <= int <= 2
    state_action = (state[0], state[1], action)
    x = tile_encode(state_action, high, low, tiling_specs, True)
    return x * weights

def epsilon_greedy(state, actions, Q, weights, high, low, tiling_specs, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    else :
        action_values = [(action, sum(Q(state, action, weights, high, low, tiling_specs))) for action in actions]
        max_action_value = None
        max_actions = []
        for a, v in action_values:
            if max_action_value == None or v > max_action_value:
                max_action_value = v
                max_actions = [a]
            elif v == max_action_value:
                max_actions.append(a)
#         print(action_values)
        return random.choice(max_actions)

# Test with some sample values
# samples = [(-0.2 , 0.067, 1)]
# TILINGS = 8
# tiling_specs = [((TILINGS, TILINGS, 3), (-0.15, -0.015, 0)),
#             ((TILINGS, TILINGS, 3), (0.0, 0.0, 0)),
#             ((TILINGS, TILINGS, 3), (0.15, 0.015, 0))]
# low = [-1.2,  -0.07, 0]
# high = [0.6, 0.07, 2]
# encoded_samples = [tile_encode(sample, high, low, tiling_specs, True) for sample in samples]
# print("\nSamples:", repr(samples), sep="\n")
# print("\nEncoded samples:", repr(encoded_samples), sep="\n")

import numpy as np
import gym
from operator import mul
from tqdm import tqdm
import pickle

env = gym.make('MountainCar-v0')
env._max_episode_steps = 2000

EPISODES = 1000
NUM_LOGS = 10

Q = action_value_approx
DELTA_Q = action_value_delta
POLICY = epsilon_greedy

ALPHA = 0.5 / 4
EPSILON = 0
GAMMA = 1
ACTIONS = [0, 1, 2]

TILES = (8, 8, 3)
TILINGS = 8
MIN_TILE_OFFSET = (-0.15, -0.015, 0)
MAX_TILE_OFFSET = (0.15, 0.015, 0)
LOW = (-1.2, -0.07, 0)
HIGH = (1.2, 0.07, 2)

val_file_name = "grad-sarsa-out/mc"
with open("{}-weights.pickle".format(val_file_name), 'rb') as handle:
    w = pickle.load(handle)

print(w)
tiling_specs = [(TILES, tuple(min_off + (max_off - min_off)*i/(TILINGS-1 if TILINGS > 1 else 1)
                                          for min_off, max_off in zip(MIN_TILE_OFFSET, MAX_TILE_OFFSET))) 
                for i in range(TILINGS)]

total_reward = 0
frames = []

s = env.reset()
a = POLICY(s, ACTIONS, Q, w, HIGH, LOW, tiling_specs, EPSILON)
done = False
episode_reward = 0
step = 0
while not done:
    frames.append(env.render(mode = 'rgb_array'))
    s_prime, r, done, info = env.step(a)
    a_prime = POLICY(s_prime, ACTIONS, Q, w, HIGH, LOW, tiling_specs, EPSILON)
    s = s_prime
    a = a_prime
    
    episode_reward += r
    total_reward += r
    step += 1

for i in range(50):
    frames.append(env.render(mode = 'rgb_array'))
    s_prime, r, done, info = env.step(a)
    a_prime = POLICY(s_prime, ACTIONS, Q, w, HIGH, LOW, tiling_specs, EPSILON)
    s = s_prime
    a = a_prime

print(episode_reward)
env.close()
with open("{}-frames.pickle".format(val_file_name), 'wb') as handle:
    pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
