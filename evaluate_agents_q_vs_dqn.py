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
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

def evaluate_q_vs_dqn(num_games=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 7  # Number of possible actions in Connect4
    state_dim = 6 * 7  # Board dimensions

    # Initialize agents
    q_agent = QLearningAgent()
    dqn_agent = DQNAgent(state_dim, action_dim)

    # Load trained models
    try:
        dqn_agent.model.load_state_dict(torch.load("agent1_model.pth", map_location=device))
        logger.info("DQN model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load DQN model: {e}")

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
        "draws": 0
    }

    # Define colors for agents
    agent_colors = {
        "Q-Learning Agent": 1,    # Represented by 1 on the board
        "DQN Agent": 2           # Represented by 2 on the board
    }

    all_games = []  # To store all games for review

    for game in range(num_games):
        env = Connect4()
        env.reset()
        state = env.board.copy()
        done = False

        game_moves = []  # To record moves of the game

        agent1_turn = (game % 2 == 0)  # Alternate which agent goes first

        while not done:
            current_agent_name = "Q-Learning Agent" if agent1_turn else "DQN Agent"
            current_agent = q_agent if agent1_turn else dqn_agent
            agent_marker = agent_colors[current_agent_name]

            if isinstance(current_agent, DQNAgent):
                action = current_agent.act(state.flatten())
            elif isinstance(current_agent, QLearningAgent):
                action = current_agent.choose_action(state, epsilon=0)  # Set epsilon=0 for evaluation
            else:
                raise ValueError(f"Unknown agent type: {type(current_agent)}")

            valid = env.make_move(action, agent_marker)
            if not valid:
                done = True
                winner = "DQN Agent" if agent1_turn else "Q-Learning Agent"
                break

            # Record the move
            move_record = (current_agent_name, action, env.board.copy())
            game_moves.append(move_record)

            winner_id = env.check_winner()
            if winner_id != 0:
                done = True
                winner = current_agent_name
            elif env.is_full():
                done = True
                winner = "draw"

            state = env.board.copy()
            agent1_turn = not agent1_turn  # Switch turns

        # Save the game moves
        all_games.append(game_moves)

        # Update results
        if winner == "Q-Learning Agent":
            results["q_agent_wins"] += 1
        elif winner == "DQN Agent":
            results["dqn_agent_wins"] += 1
        else:
            results["draws"] += 1

    # Save all games to a file
    with open("evaluated_games_q_vs_dqn.pkl", "wb") as f:
        pickle.dump(all_games, f)
    logger.info("Saved all evaluated games to 'evaluated_games_q_vs_dqn.pkl'")

    # Print results
    logger.info(f"Results after {num_games} games between Q-Learning Agent and DQN Agent:")
    logger.info(f"Q-Learning Agent Wins: {results['q_agent_wins']}")
    logger.info(f"DQN Agent Wins: {results['dqn_agent_wins']}")
    logger.info(f"Draws: {results['draws']}")

def evaluate_dqn_vs_alphazero(num_games=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 7  # Number of possible actions in Connect4
    state_dim = 6 * 7  # Board dimensions

    # Initialize agents
    dqn_agent = DQNAgent(state_dim, action_dim)
    alphazero_agent = AlphaZeroAgent(state_dim=state_dim, action_dim=action_dim, use_gpu=device.type == 'cuda')

    # Load trained models
    try:
        dqn_agent.model.load_state_dict(torch.load("dqn_model.pth", map_location=device))
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

    # Initialize results dictionary
    results = {
        "dqn_agent_wins": 0,
        "alphazero_agent_wins": 0,
        "draws": 0
    }

    # Define colors for agents
    agent_colors = {
        "DQN Agent": 2,           # Represented by 2 on the board
        "AlphaZero Agent": 3      # Represented by 3 on the board
    }

    all_games = []  # To store all games for review

    for game in range(num_games):
        env = Connect4()
        env.reset()
        state = env.board.copy()
        done = False

        game_moves = []  # To record moves of the game

        agent1_turn = (game % 2 == 0)  # Alternate which agent goes first

        while not done:
            current_agent_name = "DQN Agent" if agent1_turn else "AlphaZero Agent"
            current_agent = dqn_agent if agent1_turn else alphazero_agent
            agent_marker = agent_colors[current_agent_name]

            if isinstance(current_agent, AlphaZeroAgent):
                action, _ = current_agent.act(state, env, num_simulations=1000)
            elif isinstance(current_agent, DQNAgent):
                action = current_agent.act(state.flatten())
            else:
                raise ValueError(f"Unknown agent type: {type(current_agent)}")

            valid = env.make_move(action, agent_marker)
            if not valid:
                done = True
                winner = "AlphaZero Agent" if agent1_turn else "DQN Agent"
                break

            # Record the move
            move_record = (current_agent_name, action, env.board.copy())
            game_moves.append(move_record)

            winner_id = env.check_winner()
            if winner_id != 0:
                done = True
                winner = current_agent_name
            elif env.is_full():
                done = True
                winner = "draw"

            state = env.board.copy()
            agent1_turn = not agent1_turn  # Switch turns

        # Save the game moves
        all_games.append(game_moves)

        # Update results
        if winner == "DQN Agent":
            results["dqn_agent_wins"] += 1
        elif winner == "AlphaZero Agent":
            results["alphazero_agent_wins"] += 1
        else:
            results["draws"] += 1

    # Save all games to a file
    with open("dqn_vs_alphazero_games.pkl", "wb") as f:
        pickle.dump(all_games, f)
    logger.info("Saved all evaluated games to 'dqn_vs_alphazero_games.pkl'")

    # Print results
    logger.info(f"Results after {num_games} games between DQN Agent and AlphaZero Agent:")
    logger.info(f"DQN Agent Wins: {results['dqn_agent_wins']}")
    logger.info(f"AlphaZero Agent Wins: {results['alphazero_agent_wins']}")
    logger.info(f"Draws: {results['draws']}")

def evaluate_alphazero_vs_q(num_games=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 7  # Number of possible actions in Connect4
    state_dim = 6 * 7  # Board dimensions

    # Initialize agents
    q_agent = QLearningAgent()
    alphazero_agent = AlphaZeroAgent(state_dim=state_dim, action_dim=action_dim, use_gpu=device.type == 'cuda')

    # Load trained models
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
        "alphazero_agent_wins": 0,
        "q_agent_wins": 0,
        "draws": 0
    }

    # Define colors for agents
    agent_colors = {
        "AlphaZero Agent": 3,      # Represented by 3 on the board
        "Q-Learning Agent": 1     # Represented by 1 on the board
    }

    all_games = []  # To store all games for review

    for game in range(num_games):
        env = Connect4()
        env.reset()
        state = env.board.copy()
        done = False

        game_moves = []  # To record moves of the game

        agent1_turn = (game % 2 == 0)  # Alternate which agent goes first

        while not done:
            current_agent_name = "AlphaZero Agent" if agent1_turn else "Q-Learning Agent"
            current_agent = alphazero_agent if agent1_turn else q_agent
            agent_marker = agent_colors[current_agent_name]

            if isinstance(current_agent, AlphaZeroAgent):
                action, _ = current_agent.act(state, env, num_simulations=1000)
            elif isinstance(current_agent, QLearningAgent):
                action = current_agent.choose_action(state, epsilon=0)  # Set epsilon=0 for evaluation
            else:
                raise ValueError(f"Unknown agent type: {type(current_agent)}")

            valid = env.make_move(action, agent_marker)
            if not valid:
                done = True
                winner = "Q-Learning Agent" if agent1_turn else "AlphaZero Agent"
                break

            # Record the move
            move_record = (current_agent_name, action, env.board.copy())
            game_moves.append(move_record)

            winner_id = env.check_winner()
            if winner_id != 0:
                done = True
                winner = current_agent_name
            elif env.is_full():
                done = True
                winner = "draw"

            state = env.board.copy()
            agent1_turn = not agent1_turn  # Switch turns

        # Save the game moves
        all_games.append(game_moves)

        # Update results
        if winner == "AlphaZero Agent":
            results["alphazero_agent_wins"] += 1
        elif winner == "Q-Learning Agent":
            results["q_agent_wins"] += 1
        else:
            results["draws"] += 1

    # Save all games to a file
    with open("evaluated_games_alphazero_vs_q.pkl", "wb") as f:
        pickle.dump(all_games, f)
    logger.info("Saved all evaluated games to 'evaluated_games_alphazero_vs_q.pkl'")

    # Print results
    logger.info(f"Results after {num_games} games between AlphaZero Agent and Q-Learning Agent:")
    logger.info(f"AlphaZero Agent Wins: {results['alphazero_agent_wins']}")
    logger.info(f"Q-Learning Agent Wins: {results['q_agent_wins']}")
    logger.info(f"Draws: {results['draws']}")

if __name__ == "__main__":
    evaluate_q_vs_dqn(num_games=100)
    evaluate_dqn_vs_alphazero(num_games=100)
    evaluate_alphazero_vs_q(num_games=100)