import pickle

with open("q_agent_q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

print(f"Number of state-action pairs in Q-table: {len(q_table)}")