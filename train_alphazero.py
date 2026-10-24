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

def simulate_game(agent, time_limit, action_dim, checkpoint_interval=10):
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
                action, action_probs = agent.act(state, env, num_simulations=1000)

                # Validate action_probs length
                if len(action_probs) != action_dim:
                    logger.error(
                        f"action_probs length {len(action_probs)} does not match action_dim {action_dim}."
                    )
                    break

                # Validate action
                legal_moves = env.get_legal_moves()
                if action not in legal_moves:
                    logger.error(f"Agent selected an illegal move: {action}")
                    break

                # Make the move
                env.make_move(action)
                states.append(state.copy())
                mcts_probs.append(action_probs)
                values.append(0)  # Placeholder; update with actual game result later

                # Check for game end
                winner_id = env.check_winner()
                if winner_id != 0:
                    done = True
                    result = env.game_result()
                    values = [result for _ in states]  # Update all values
                elif env.is_full():
                    done = True
                    result = 0  # Draw
                    values = [result for _ in states]  # Update all values

                # Update state
                state = env.board.copy()

            # Check consistency before training
            if len(states) == len(mcts_probs) == len(values) > 0:
                # Convert to arrays
                states = torch.tensor(states, dtype=torch.float32)
                mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32)
                values = torch.tensor(values, dtype=torch.float32)

                print(f"states shape: {states.shape}")
                print(f"mcts_probs shape: {mcts_probs.shape}")
                print(f"values shape: {values.shape}")

                # Train the agent
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

            episode += 1
            logger.info(f"Moving to Episode {episode}")

        logger.info("Simulation finished.")
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

    simulate_game(agent, time_limit, action_dim)

if __name__ == "__main__":
    # Example usage:
    # python train_alphazero.py
    train_alphazero(time_limit=6*60*60, load_model=True, checkpoint_path=None)