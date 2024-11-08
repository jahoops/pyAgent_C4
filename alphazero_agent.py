# alphazero_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import logging
from connect4 import Connect4, Connect4Net  # Correct imports
from torch.utils.data import DataLoader  # Add DataLoader import

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env  # Current Connect4 environment
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.env.get_legal_moves())

    def best_child(self, exploration_constant=1.414):
        """
        Selects the child with the highest UCB score.
        """
        best_score = -float('inf')
        best_child = None
        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb_score = float('inf')
            else:
                ucb_score = (child.value_sum / child.visit_count) + \
                            exploration_constant * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child

    def expand(self, action_probs):
        """
        Expands the node by adding child nodes for each legal action with their prior probabilities.
        """
        for action, prob in zip(self.env.get_legal_moves(), action_probs):
            if action not in self.children:
                new_env = self.env.clone()
                new_env.make_move(action)
                self.children[action] = MCTSNode(new_env, parent=self, action=action)
                self.children[action].prior = prob

class AlphaZeroAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, use_gpu=True):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Initialize the model and move it to the device
        self.model = Connect4Net(action_dim).to(self.device)

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.temperature = 1  # Initial temperature for exploration

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            logger.info("Model loaded successfully.")
        except RuntimeError as e:
            logger.error(f"Error loading state_dict: {e}")

    def preprocess(self, board):
        """
        Preprocesses the board state for the neural network.
        Converts the board to a tensor and moves it to the device.
        Expected shape: [1, 1, 6, 7]
        """
        board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0).to(self.device)
        return board_tensor

    def act(self, state, env, num_simulations=800):
        # Perform MCTS simulations
        actions, probabilities = self.mcts_simulate(state, env, num_simulations)

        # Create action_probs array with zeros
        action_probs = [0.0] * self.action_dim

        # Assign probabilities to the corresponding actions
        for action, probability in zip(actions, probabilities):
            action_probs[action] = probability

        # Select the action
        if self.temperature == 0:
            # Greedy action selection
            selected_action = actions[np.argmax(probabilities)]
        else:
            # Stochastic action selection
            selected_action = np.random.choice(actions, p=probabilities)

        return selected_action, action_probs

    def mcts_simulate(self, state, env, num_simulations=800):
        root = MCTSNode(env.clone())
        #logger.debug("Starting MCTS simulations.")

        log_interval = 100
        for simulation in range(1, num_simulations + 1):
            node = root
            env_copy = env.clone()

            # Selection
            while node.is_fully_expanded() and not env_copy.is_terminal():
                node = node.best_child()
                if node is None:
                    logger.error(f"best_child returned None at simulation {simulation}.")
                    break
                env_copy.make_move(node.action)

            if node is None:
                logger.error(f"No valid child to expand at simulation {simulation}. Skipping.")
                continue  # Skip to next simulation

            # Expansion
            if not node.is_fully_expanded() and not env_copy.is_terminal():
                legal_moves = env_copy.get_legal_moves()
                if not legal_moves:
                    logger.debug("No legal moves available for expansion.")
                    continue  # Skip expansion if no legal moves

                state_tensor = self.preprocess(env_copy.board)
                try:
                    log_policy, _ = self.model(state_tensor)
                    policy = torch.exp(log_policy).cpu().detach().numpy().flatten()
                except Exception as e:
                    logger.error(f"Error during model inference: {e}")
                    continue  # Skip this simulation

                # Mask illegal moves
                filtered_probs = np.zeros(self.action_dim)
                filtered_probs[legal_moves] = policy[legal_moves]
                if filtered_probs.sum() > 0:
                    filtered_probs /= filtered_probs.sum()
                else:
                    if len(legal_moves) > 0:
                        filtered_probs[legal_moves] = 1.0 / len(legal_moves)
                    else:
                        logger.error("No legal moves to assign probabilities.")
                        continue  # Skip if no legal moves

                node.expand(filtered_probs)
                #logger.debug(f"Expanded node with actions: {legal_moves}")

            # Simulation
            leaf_node = node
            if leaf_node is None:
                logger.error(f"Leaf node is None at simulation {simulation}.")
                continue  # Safety check
            env_leaf = env_copy.clone()  # Clone env_copy for simulation
            state_tensor = self.preprocess(env_leaf.board)
            try:
                with torch.no_grad():
                    _, value = self.model(state_tensor)
                value = value.item()
            except Exception as e:
                logger.error(f"Error during simulation forward pass: {e}")
                continue

            # Backpropagation
            while leaf_node is not None:
                leaf_node.visit_count += 1
                leaf_node.value_sum += value
                leaf_node = leaf_node.parent

            # Periodic Logging
            #if simulation % log_interval == 0:
                #logger.debug(f"Completed {simulation} simulations.")

        # Action Selection
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())

        # Check if there are available actions
        if not actions:
            logger.error("No actions available after MCTS. Returning default values.")
            # Return default values or handle the error appropriately
            return [], []

        # Normalize visit counts to get probabilities
        total_counts = np.sum(visit_counts)
        probabilities = visit_counts / total_counts if total_counts > 0 else np.zeros_like(visit_counts)

        return actions, probabilities

    def train(self, states, mcts_probs, values):
        # Convert lists to tensors
        states = torch.tensor(states, dtype=torch.float32)  # Shape: [batch_size, 6, 7]
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        # Add channel dimension to states
        states = states.unsqueeze(1)  # Now shape: [batch_size, 1, 6, 7]

        # Move data to the appropriate device
        states = states.to(self.device)
        mcts_probs = mcts_probs.to(self.device)
        values = values.to(self.device)

        # Forward pass
        log_policy, predicted_values = self.model(states)

        # Compute losses
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_policy, dim=1))
        value_loss = torch.mean((predicted_values.squeeze() - values) ** 2)
        total_loss = policy_loss + value_loss

        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()