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

import gym

ACTIONS = [0, 1]
OBS_SPACE_BUCKETS = 10

Q = {}

EPISODES = 1
NUM_LOGS = 1
ALPHA = 0.5
EPSILON = 0.1
GAMMA = 1

policy = epsilon_greedy
environment = gym.make('CartPole-v1')

total_reward = 0

print("Starting training for {} steps".format(EPISODES))

for i in range(0, EPISODES):

    s = Observation(environment.reset())
    a = policy(s, Q, ACTIONS, EPSILON)
    done = False


    while not done:
        s_prime, r, done, info = environment.step(a)
        s_prime = Observation(s_prime)
        a_prime = policy(s_prime, Q, ACTIONS, EPSILON)
        
        Q[(s, a)] = Q.get((s, a), 0) + ALPHA * (r + GAMMA * Q.get((s_prime, a_prime), 0) - Q.get((s, a), 0))
        
        s = s_prime
        a = a_prime
        
        total_reward += r
    
    if i % (EPISODES // NUM_LOGS) == 0:
        print("Episode {} with average reward {}".format(i, total_reward / (EPISODES // NUM_LOGS)))
        total_reward = 0

environment.close()

val_file_name = "sarsa-out/sarsa-CartPole"
print("Training Complete - saving value dict to {}".format(val_file_name))
import pickle
with open("{}-Q.pickle".format(val_file_name), 'wb') as handle:
    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

print("Validating results and generating visualisation")
valid_env = gym.make('CartPole-v0')
valid_s = Observation(valid_env.reset())
valid_a = policy(s, Q, ACTIONS, EPSILON)
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

    valid_reward += r
    valid_timesteps += 1
#     print(valid_a)

print(valid_reward)
valid_env.close()

with open("{}-frames.pickle".format(val_file_name), 'wb') as handle:
    pickle.dump(valid_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
