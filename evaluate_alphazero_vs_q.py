# evaluate_alphazero_vs_q.py
from connect4 import Connect4
from q_learning_agent import QLearningAgent
from alphazero_agent import AlphaZeroAgent
import torch
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def evaluate_alphazero_vs_q(num_games=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 7
    state_dim = 6 * 7

    q_agent = QLearningAgent()
    alphazero_agent = AlphaZeroAgent(state_dim=state_dim, action_dim=action_dim, use_gpu=device.type == 'cuda')

    try:
        alphazero_agent.load_model("alphazero_model_final.pth")
    except Exception as e:
        logger.error(f"Failed to load AlphaZero model: {e}")

    try:
        with open("q_agent_q_table.pkl", "rb") as f:
            q_agent.q_table = pickle.load(f)
        logger.info("Q-Learning Q-table loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Q-Learning Q-table: {e}")

    results = {"alphazero_agent_wins": 0, "q_agent_wins": 0, "draws": 0}
    agent_colors = {"AlphaZero Agent": 3, "Q-Learning Agent": 1}
    all_games = []

    for game in range(num_games):
        env = Connect4()
        env.reset()
        state = env.board.copy()
        done = False
        game_moves = []
        agent1_turn = (game % 2 == 0)

        while not done:
            current_agent_name = "AlphaZero Agent" if agent1_turn else "Q-Learning Agent"
            current_agent = alphazero_agent if agent1_turn else q_agent
            agent_marker = agent_colors[current_agent_name]

            if isinstance(current_agent, AlphaZeroAgent):
                action, _ = current_agent.act(state, env, num_simulations=500)
            elif isinstance(current_agent, QLearningAgent):
                action = current_agent.choose_action(state, epsilon=0)
            else:
                raise ValueError(f"Unknown agent type: {type(current_agent)}")

            valid = env.make_move(action, agent_marker)
            if not valid:
                done = True
                winner = "Q-Learning Agent" if agent1_turn else "AlphaZero Agent"
                break

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
            agent1_turn = not agent1_turn

        all_games.append(game_moves)
        if winner == "AlphaZero Agent":
            results["alphazero_agent_wins"] += 1
        elif winner == "Q-Learning Agent":
            results["q_agent_wins"] += 1
        else:
            results["draws"] += 1

    with open("alphazero_vs_q_games.pkl", "wb") as f:
        pickle.dump(all_games, f)
    logger.info("Saved all evaluated games to 'alphazero_vs_q_games.pkl'")

    logger.info(f"Results after {num_games} games between AlphaZero Agent and Q-Learning Agent:")
    logger.info(f"AlphaZero Agent Wins: {results['alphazero_agent_wins']}")
    logger.info(f"Q-Learning Agent Wins: {results['q_agent_wins']}")
    logger.info(f"Draws: {results['draws']}")

if __name__ == "__main__":
    evaluate_alphazero_vs_q(num_games=100)