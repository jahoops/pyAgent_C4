# q_learning_agent.py
import numpy as np
import random
from connect4 import Connect4  # Ensure this import matches your project structure

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor

    def get_state_key(self, state):
        # Ensure state is a NumPy array
        if isinstance(state, Connect4):
            state = state.board
        return tuple(state.flatten())

    def choose_action(self, state, epsilon=0.1):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table or random.random() < epsilon:
            # Choose a random action
            action = random.choice(range(7))  # Assuming 7 columns
        else:
            # Choose the action with the highest Q-value
            action_values = self.q_table[state_key]
            action = np.argmax(action_values)
        return action

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(7)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(7)
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error

# No top-level code here