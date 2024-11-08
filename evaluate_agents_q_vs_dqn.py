# evaluate_agents_q_vs_dqn.py

import logging
from connect4 import Connect4
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
from alphazero_agent import AlphaZeroAgent
import torch
import pickle
import random
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def evaluate_agents(num_games=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 7  # Number of possible actions in Connect4
    state_dim = 6 * 7  # Board dimensions

    # Initialize agents
    q_agent = QLearningAgent()
    dqn_agent = DQNAgent(state_dim, action_dim)
    alphazero_agent = AlphaZeroAgent(state_dim=state_dim, action_dim=action_dim, use_gpu=device.type == 'cuda')

    # Load trained models
    try:
        dqn_agent.model.load_state_dict(torch.load("agent1_model.pth", map_location=torch.device('cpu')))
        logger.info("DQN model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load DQN model: {e}")

    try:
        alphazero_agent.load_model("alphazero_model_final.pth")
    except Exception as e:
        logger.error(f"Failed to load AlphaZero model: {e}")

    # Verify if AlphaZero model is loaded
    if not hasattr(alphazero_agent.model, 'forward'):
        logger.error("AlphaZero model failed to load properly. Exiting evaluation.")
        return

    # Load Q-table for Q-Learning agent
    try:
        with open("q_agent_q_table.pkl", "rb") as f:
            q_agent.q_table = pickle.load(f)
        logger.info("Q-Learning Q-table loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Q-Learning Q-table: {e}")

    # Initialize results dictionary
    results = {
        "q_agent_wins": 0,
        "dqn_agent_wins": 0,
        "alphazero_agent_wins": 0,
        "draws": 0
    }

    # Define mapping from agent names to result keys
    agent_result_keys = {
        "Q-Learning Agent": "q_agent_wins",
        "DQN Agent": "dqn_agent_wins",
        "AlphaZero Agent": "alphazero_agent_wins",
    }

    # Define matchups
    matchups = [
        ("Q-Learning Agent", q_agent, "DQN Agent", dqn_agent),
        ("DQN Agent", dqn_agent, "AlphaZero Agent", alphazero_agent),
        ("AlphaZero Agent", alphazero_agent, "Q-Learning Agent", q_agent)
    ]

    for matchup in matchups:
        agent1_name, agent1, agent2_name, agent2 = matchup
        logger.info(f"Evaluating {agent1_name} vs {agent2_name}")
        for game in range(num_games):
            env = Connect4()
            env.reset()
            state = env.board.copy()
            done = False
            states, mcts_probs, values = [], [], []

            agent1_turn = (game % 2 == 0)  # Alternate which agent goes first

            while not done:
                if agent1_turn:
                    if isinstance(agent1, AlphaZeroAgent):
                        action = agent1.act(state, env, num_simulations=100)  # Adjust simulations as needed
                    elif isinstance(agent1, DQNAgent):
                        action = agent1.act(state.flatten())
                    elif isinstance(agent1, QLearningAgent):
                        action = agent1.choose_action(state, epsilon=0)  # Set epsilon=0 for evaluation
                    else:
                        raise ValueError(f"Unknown agent type: {type(agent1)}")
                else:
                    if isinstance(agent2, AlphaZeroAgent):
                        action = agent2.act(state, env, num_simulations=100)  # Adjust simulations as needed
                    elif isinstance(agent2, DQNAgent):
                        action = agent2.act(state.flatten())
                    elif isinstance(agent2, QLearningAgent):
                        action = agent2.choose_action(state, epsilon=0)  # Set epsilon=0 for evaluation
                    else:
                        raise ValueError(f"Unknown agent type: {type(agent2)}")

                valid = env.make_move(action)
                if not valid:
                    done = True
                    winner = agent2_name if agent1_turn else agent1_name
                    break

                winner_id = env.check_winner()
                if winner_id != 0:
                    done = True
                    winner = agent1_name if (winner_id == 1 and agent1_turn) else agent2_name
                elif env.is_full():
                    done = True
                    winner = "draw"

                state = env.board.copy()
                agent1_turn = not agent1_turn  # Switch turns

            if winner == agent1_name:
                results_key = agent_result_keys.get(agent1_name)
                if results_key:
                    results[results_key] += 1
            elif winner == agent2_name:
                results_key = agent_result_keys.get(agent2_name)
                if results_key:
                    results[results_key] += 1
            else:
                results["draws"] += 1

        # Print results for this matchup
        logger.info(f"Results after {num_games} games between {agent1_name} and {agent2_name}:")
        logger.info(f"{agent1_name} Wins: {results[agent_result_keys[agent1_name]]}")
        logger.info(f"{agent2_name} Wins: {results[agent_result_keys[agent2_name]]}")
        logger.info(f"Draws: {results['draws']}\n")

        # Reset results for next matchup
        results = {key: 0 for key in results}

    logger.info(f"Completed evaluation of {num_games} games per matchup.")

if __name__ == "__main__":
    evaluate_agents(num_games=100)