import os
import sys

# Ensure 'python' directory is on sys.path
ROOT_PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PY_DIR not in sys.path:
    sys.path.append(ROOT_PY_DIR)

from models.dqn_agent import DQNAgent

def main():
    agent = DQNAgent(
        state_dim=10,
        action_dim=3,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=1000,
        batch_size=32,
        target_update_freq=100,
        hidden_layers=[64, 64],
    )
    print(f"Agent device: {agent.device}")

if __name__ == "__main__":
    main()