import random

ROUND_TO = 1

class Observation:
    
    def __init__(self, state):
        self.cart_pos = round(state[0], ROUND_TO)
        self.cart_velocity = round(state[1], ROUND_TO)
        self.pole_pos = round(state[2], ROUND_TO)
        self.pole_velocity = round(state[3], ROUND_TO)
        
    def __repr__(self):
        return "({}, {}, {}, {})".format(self.cart_pos, self.cart_velocity, self.pole_pos, self.pole_velocity)
    
    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        return (self.cart_pos == other.cart_pos
            and self.cart_velocity == other.cart_velocity
            and self.pole_pos == other.pole_pos
            and self.pole_velocity == other.pole_velocity)
            
def epsilon_greedy(state, q_values, actions, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    
    
    current_max = None
    current_actions = []

    for a in actions:
        val = q_values.get((state, a), 0)
        if current_max == None or val > current_max:
            current_max = val
            current_actions = [a]
        elif val == current_max:
            current_actions.append(a)
    
    return random.choice(current_actions)

import pickle
import gym

ACTIONS = [0, 1]
OBS_SPACE_BUCKETS = 10

Q = {}

EPISODES = 10000
NUM_LOGS = 20
ALPHA = 0.5
EPSILON = 0.1
GAMMA = 1

policy = epsilon_greedy

val_file_name = "sarsa-out/sarsa-CartPole"
with open("{}-Q.pickle".format(val_file_name), 'rb') as handle:
    Q = pickle.load(handle)
    

print("Validating results and generating visualisation")
valid_env = gym.make('CartPole-v0')
valid_s = Observation(valid_env.reset())
valid_a = policy(valid_s, Q, ACTIONS, EPSILON)
valid_done = False
valid_reward = 0
valid_timesteps = 0
valid_frames = []

while not valid_done:
    valid_frames.append(valid_env.render(mode = 'rgb_array'))
    valid_s_prime, valid_r, valid_done, valid_info = valid_env.step(valid_a)
    valid_s_prime = Observation(valid_s_prime)
    valid_a_prime = policy(valid_s_prime, Q, ACTIONS, EPSILON)

    valid_s = valid_s_prime
    valid_a = valid_a_prime

    valid_reward += valid_r
    valid_timesteps += 1

for i in range(50):
    valid_frames.append(valid_env.render(mode = 'rgb_array'))
    valid_s_prime, valid_r, valid_done, valid_info = valid_env.step(valid_a)
    valid_s_prime = Observation(valid_s_prime)
    valid_a_prime = policy(valid_s_prime, Q, ACTIONS, EPSILON)

    valid_s = valid_s_prime
    valid_a = valid_a_prime

    valid_reward += valid_r
    valid_timesteps += 1

print(valid_reward)
valid_env.close()

with open("{}-frames.pickle".format(val_file_name), 'wb') as handle:
    pickle.dump(valid_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
