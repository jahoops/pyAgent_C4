# train_alphazero.py
import logging
import time
import copy
from connect4 import Connect4
from alphazero_agent import AlphaZeroAgent, MCTSNode
import torch

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# train_alphazero.py

def simulate_game(agent, time_limit, checkpoint_interval=10):
    try:
        logger.info("Simulation started.")
        start_time = time.time()
        episode = 0

        while (time.time() - start_time) < time_limit:
            env = Connect4()
            env.reset()
            state = env.board.copy()
            done = False
            states, mcts_probs, values = [], [], []

            while not done:
                # Agent's Turn
                action, action_probs = agent.act(state, env, num_simulations=800)
                if action == -1:
                    logger.error("Agent returned an invalid action.")
                    break

                legal_moves = env.get_legal_moves()
                if action not in legal_moves:
                    logger.error(f"Agent selected an illegal move: {action}")
                    break

                env.make_move(action)
                states.append(state.copy())

                # Placeholder: Assigning uniform probabilities; replace with actual MCTS probabilities
                action_probs = [0] * agent.action_dim
                action_probs[action] = 1.0
                mcts_probs.append(action_probs)

                # Check for game end
                winner_id = env.check_winner()
                if winner_id != 0:
                    done = True
                    result = env.game_result()  # Correctly call game_result on env
                    values = [result for _ in states]  # Assign to all states
                elif env.is_full():
                    done = True
                    result = 0  # Draw
                    values = [result for _ in states]  # Assign to all states

                state = env.board.copy()

            # Train the agent with the collected data
            if len(states) == len(mcts_probs) == len(values) > 0:
                agent.train(states, mcts_probs, values)
                logger.info(f"Trained on Episode {episode + 1}")
            else:
                logger.warning(f"Inconsistent data lengths at Episode {episode + 1}")

            # Periodic Checkpoint Saving
            if (episode + 1) % checkpoint_interval == 0:
                checkpoint_path = f"alphazero_checkpoint_episode_{episode + 1}.pth"
                torch.save({
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'episode': episode + 1
                }, checkpoint_path)
                logger.info(f"[SAVE] Checkpoint saved to {checkpoint_path}")

            # Periodic Logging
            if (episode + 1) % 100 == 0:
                logger.info(f"Completed {episode + 1} episodes.")

            episode += 1
            logger.info(f"Moving to Episode {episode}")

        logger.info("Simulation finished.")
    except KeyboardInterrupt:
        logger.info("\n[INFO] Training interrupted by user.")
    except Exception as e:
        logger.error(f"Simulation encountered an error: {e}")
    finally:
        # Save the final model state
        final_model_path = "alphazero_model_final.pth"
        torch.save(agent.model.state_dict(), final_model_path)
        logger.info(f"[SAVE] Final model saved to {final_model_path}.")

def train_alphazero(time_limit=1*60*60, load_model=True, checkpoint_path=None):
    action_dim = 7
    state_dim = 42  # 6 rows * 7 columns
    agent = AlphaZeroAgent(state_dim=state_dim, action_dim=action_dim, use_gpu=False)

    if load_model:
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=agent.device)
                agent.model.load_state_dict(checkpoint['model_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint.get('episode', 0)
                logger.info(f"Loaded checkpoint from {checkpoint_path} at Episode {start_episode}.")
            except (FileNotFoundError, KeyError) as e:
                logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
                logger.info("Starting training from scratch.")
        else:
            try:
                agent.load_model("alphazero_model_final.pth")
                logger.info("Loaded model from alphazero_model.pth successfully.")
            except RuntimeError as e:
                logger.error(f"Failed to load model from alphazero_model_final.pth: {e}")
                logger.info("Starting training from scratch.")
            except FileNotFoundError:
                logger.warning("alphazero_model_final.pth not found. Starting training from scratch.")

    simulate_game(agent, time_limit)

if __name__ == "__main__":
    # Example usage:
    # python train_alphazero.py
    train_alphazero(time_limit=6*60*60, load_model=True, checkpoint_path=None)