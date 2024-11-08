# test_mctsnode.py
import numpy as np

from connect4 import Connect4
from alphazero_agent import MCTSNode

def test_mctsnode_expand():
    env = Connect4()
    root = MCTSNode(env)
    action_probs = np.array([0.1, 0.2, 0.3, 0.15, 0.15, 0.05, 0.05])
    root.expand(action_probs)
    assert len(root.children) == len(env.get_legal_moves()), "Expand method failed to add all children"
    print("MCTSNode expand method works correctly.")

if __name__ == "__main__":
    test_mctsnode_expand()