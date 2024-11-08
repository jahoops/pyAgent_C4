# train_agents.py
from connect4 import Connect4
from q_learning_agent import QLearningAgent
import pickle
import multiprocessing as mp
import time

def train_agent(agent, time_limit, return_dict, idx):
    env = Connect4()
    start_time = time.time()
    episode = 0

    while time.time() - start_time < time_limit:
        state = env.reset()
        done = False
        agent1_turn = (episode % 2 == 0)  # Alternate which agent goes first

        while not done:
            valid_actions = [c for c in range(7) if env.board[0, c] == 0]
            if agent1_turn:
                # Agent 1's turn
                action = agent.choose_action(state, valid_actions)
                env.make_move(action)
                reward = 1 if env.check_winner() == 1 else 0
                next_state = env.board.copy()
                agent.update_q_value(state, action, reward, next_state, valid_actions)
            else:
                # Agent 2's turn
                action = agent.choose_action(state, valid_actions)
                env.make_move(action)
                reward = -1 if env.check_winner() == 2 else 0
                next_state = env.board.copy()
                agent.update_q_value(state, action, reward, next_state, valid_actions)

            state = next_state
            if reward == 1 or reward == -1 or env.is_full():
                done = True

            agent1_turn = not agent1_turn  # Alternate turns

        episode += 1
        # Log progress
        if episode % 1000 == 0:
            print(f"Process {idx} - Episode {episode} - Elapsed Time: {time.time() - start_time:.2f} seconds")

    return_dict[idx] = agent.q_table

def train_agents(time_limit=10*60, num_processes=4):  # Time limit in seconds (30 minutes)
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

    # Aggregate Q-tables from all processes
    final_q_table = {}
    for q_table in return_dict.values():
        for state, actions in q_table.items():
            if state not in final_q_table:
                final_q_table[state] = actions
            else:
                final_q_table[state] = (final_q_table[state] + actions) / 2  # Average the Q-values

    # Save the final Q-table
    with open("q_agent_q_table.pkl", "wb") as f:
        pickle.dump(final_q_table, f)

if __name__ == "__main__":
    train_agents()