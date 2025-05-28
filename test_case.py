import pickle
import os
import random
from Environment import NumberStabilizer

print("Testing greedy policy:")

if not os.path.exists('q_table.pkl'):
    raise FileNotFoundError("Q-table not found. Run training first!")
with open('q_table.pkl', 'rb') as f:
    Q = pickle.load(f)

actions = [-1, 1]  # Possible actions
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.0  # No exploration

stabilizer = NumberStabilizer()
state = stabilizer.reset()
done = False
total_reward = 0
steps = 0

def choose_action(state):
    q_values = [Q.get((state, a), 0) for a in actions]
    max_q = max(q_values)
    return actions[q_values.index(max_q)]

while not done and steps < 100:
    action = choose_action(state)
    next_state, reward, done = stabilizer.step(action)
    total_reward += reward
    print(f"Step {steps + 1}: State={state}, Action={action}, Next={next_state}, Reward={reward}")
    state = next_state
    steps += 1

print(f"Test completed in {steps} steps with total reward: {total_reward}")
