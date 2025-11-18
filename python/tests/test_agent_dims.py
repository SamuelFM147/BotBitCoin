import numpy as np
import pandas as pd

from environment.orderbook_env import OrderBookTradingEnv
from models.dqn_agent import DQNAgent
from models.qrdqn_agent import QRDQNAgent


def make_df(n: int = 120):
    prices = 100.0 + 0.2 * np.arange(n, dtype=np.float32)
    return pd.DataFrame({'close': prices, 'open': prices, 'high': prices * 1.001, 'low': prices * 0.999, 'volume': np.ones(n)})


def test_agent_initialization_and_action_range():
    df = make_df()
    env = OrderBookTradingEnv(df, lookback_window=25, fee_jitter_pct=0.0)
    obs, _ = env.reset(seed=123)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # DQN
    dqn = DQNAgent(state_dim=state_dim, action_dim=action_dim, buffer_size=128, batch_size=32, target_update_freq=50)
    a1 = dqn.select_action(obs, training=False)
    assert isinstance(a1, int)
    assert 0 <= a1 < action_dim

    # QRDQN
    qrdqn = QRDQNAgent(state_dim=state_dim, action_dim=action_dim, buffer_size=128, batch_size=32, target_update_freq=50)
    a2 = qrdqn.select_action(obs, training=False)
    assert isinstance(a2, int)
    assert 0 <= a2 < action_dim

