# train_agents.py
from connect4 import Connect4
from q_learning_agent import QLearningAgent
import pickle
import multiprocessing as mp
import time
import random
import numpy as np

def train_agent(agent, time_limit, return_dict, idx):
    start_time = time.time()
    episode = 0
    env = Connect4()

    while time.time() - start_time < time_limit:
        env.reset()
        state = env.board.copy()
        done = False
        agent1_turn = True

        while not done:
            if agent1_turn:
                action = agent.choose_action(state)
                valid = env.make_move(action)
                reward = -1 if env.check_winner() == 2 else 0
                next_state = env.board.copy()
                agent.update_q_value(state, action, reward, next_state)
                state = next_state
                if reward == 1 or reward == -1 or env.is_full():
                    done = True
            else:
                # Opponent's turn (random move for simplicity)
                action = random.choice(env.get_legal_moves())
                env.make_move(action)
                if env.check_winner() != 0 or env.is_full():
                    done = True

            agent1_turn = not agent1_turn  # Alternate turns

        episode += 1
        # Log progress
        if episode % 1000 == 0:
            print(f"Process {idx} - Episode {episode} - Elapsed Time: {time.time() - start_time:.2f} seconds")

    return_dict[idx] = agent.q_table

def train_agents(time_limit=1*60, num_processes=4):  # Time limit in seconds (30 minutes)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(num_processes):
        agent = QLearningAgent()
        p = mp.Process(target=train_agent, args=(agent, time_limit, return_dict, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Process {p.pid} exited with code {p.exitcode}")

    print("All processes have completed. Aggregating Q-tables...")

    # Aggregate Q-tables from all processes
    final_q_table = {}
    for q_table in return_dict.values():
        for state_key, action_values in q_table.items():
            if state_key not in final_q_table:
                final_q_table[state_key] = action_values
            else:
                final_q_table[state_key] = np.maximum(final_q_table[state_key], action_values)

    print("Q-tables aggregated. Saving to file...")

    try:
        # Save the final Q-table
        with open("q_agent_q_table.pkl", "wb") as f:
            pickle.dump(final_q_table, f)
        print("Saved final Q-table to 'q_agent_q_table.pkl'")
    except Exception as e:
        print(f"Error saving Q-table: {e}")

if __name__ == "__main__":
    train_agents()