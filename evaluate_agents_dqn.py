# evaluate_agents.py
from connect4 import Connect4
from dqn_agent import DQNAgent
import torch

def evaluate_agents(num_games=100):
    env = Connect4()
    state_dim = env.board.size
    action_dim = 7
    agent1 = DQNAgent(state_dim, action_dim)
    agent2 = DQNAgent(state_dim, action_dim)

    # Load trained models
    agent1.model.load_state_dict(torch.load("agent1_model.pth"))
    agent2.model.load_state_dict(torch.load("agent2_model.pth"))

    results = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}

    for game in range(num_games):
        state = env.reset().flatten()
        done = False
        while not done:
            action1 = agent1.act(state)
            env.make_move(action1)
            if env.check_winner() == 1:
                results["agent1_wins"] += 1
                done = True
                break
            if env.is_full():
                results["draws"] += 1
                done = True
                break

            action2 = agent2.act(state)
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