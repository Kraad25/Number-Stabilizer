import pickle
import os
import random

from Environment import NumberStabilizer

actions = [-1, 1] # Possible actions
alpha = 0.1 # Learning rate
gamma = 0.9 # Discount factor
epsilon = 0.2 # Exploration rate

if os.path.exists('q_table.pkl'):
    with open('q_table.pkl', 'rb') as f:
        Q = pickle.load(f)
else:
    Q = {}  #(state, action) -> Q-value

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        q_values = [Q.get((state, a), 0) for a in actions]
        max_q = max(q_values)
        return actions[q_values.index(max_q)]

stabilizer = NumberStabilizer()
episodes = 100
success_count = 0

for episode in range(episodes):
    state = stabilizer.reset()
    done = False
    steps = 0

    while not done and steps < 100:
        """
        Main training loop for the Q-learning agent.
        Limiting the training to 100 steps per episode is 
        important as it learns not to waste time and accumulate a lot of penalities.
        """
        action = choose_action(state)
        next_state, reward, done = stabilizer.step(action)
        if reward == 1:
            success_count += 1
        
        # Update Q-value
        old_q_value = Q.get((state, action), 0)
        next_q = [Q.get((next_state, a), 0) for a in actions]
        next_max_q = max(next_q)
        new_q_value = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)
        Q[(state, action)] = new_q_value
        
        state = next_state
        steps += 1

with open('q_table.pkl', 'wb') as f:
    pickle.dump(Q, f)

print(f"\nOut of {episodes} episodes, succeeded in {success_count} episodes.")