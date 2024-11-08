# train_dqn.py
from connect4 import Connect4
from dqn_agent import DQNAgent
import numpy as np
import torch
import time

def simulate_game(agent, time_limit, batch_size):
    try:
        print("Simulation started.")
        env = Connect4()
        start_time = time.time()
        episode = 0

        while time.time() - start_time < time_limit:
            state = env.reset().flatten()
            done = False
            agent1_turn = (episode % 2 == 0)  # Alternate which agent goes first
            move_count = 0  # Track the number of moves to prevent infinite loops

            while not done:
                if agent1_turn:
                    # Agent 1's turn
                    action = agent.act(state)
                else:
                    # Agent 2's turn
                    action = agent.act(state)

                valid = env.make_move(action)
                if not valid:
                    reward = -10
                    next_state = state
                    done = True
                else:
                    reward = 1 if env.check_winner() == 1 else 0
                    next_state = env.board.flatten()
                    done = reward == 1 or env.is_full()

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

                agent1_turn = not agent1_turn  # Alternate turns
                move_count += 1

                # Timeout mechanism to prevent infinite loops
                if move_count > 100:
                    print("Timeout: Ending game after 100 moves.")
                    done = True

            agent.update_target_model()
            print(f"Episode {episode} - Epsilon: {agent.epsilon:.2f}")
            episode += 1
            if episode % 10 == 0:  # Log progress every 10 episodes for debugging
                elapsed_time = time.time() - start_time
        print("Simulation finished.")
    except Exception as e:
        print(f"Simulation encountered an error: {e}")

def train_dqn(time_limit=90*60, batch_size=64, load_model=True):  # Time limit in seconds (30 minutes)
    state_dim = 42  # 6 rows * 7 columns
    action_dim = 7
    agent = DQNAgent(state_dim, action_dim)

    if load_model:
        try:
            agent.model.load_state_dict(torch.load("agent1_model.pth"))
            print("Loaded saved model.")
        except FileNotFoundError:
            print("No saved model found. Starting training from scratch.")

    simulate_game(agent, time_limit, batch_size)



    # Save the trained model
    torch.save(agent.model.state_dict(), "agent1_model.pth")

if __name__ == "__main__":
    start_time = time.time()
    train_dqn()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")