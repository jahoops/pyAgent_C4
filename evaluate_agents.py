# evaluate_agents.py
from connect4 import Connect4
from q_learning_agent import QLearningAgent

def evaluate_agents(num_games=100):
    env = Connect4()
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()

    agent1.q_table = ...  # Load trained Q-table for agent1
    agent2.q_table = ...  # Load trained Q-table for agent2

    results = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}

    for game in range(num_games):
        state = env.reset()
        done = False
        while not done:
            valid_actions = [c for c in range(7) if env.board[0, c] == 0]
            action1 = agent1.choose_action(state, valid_actions)
            env.make_move(action1)
            if env.check_winner() == 1:
                results["agent1_wins"] += 1
                done = True
                break
            if env.is_full():
                results["draws"] += 1
                done = True
                break

            action2 = agent2.choose_action(state, valid_actions)
            env.make_move(action2)
            if env.check_winner() == 2:
                results["agent2_wins"] += 1
                done = True
                break
            if env.is_full():
                results["draws"] += 1
                done = True

    print(results)

if __name__ == "__main__":
    evaluate_agents()