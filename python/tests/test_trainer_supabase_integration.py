import os
from unittest import mock

import numpy as np
import pandas as pd

from training.trainer import Trainer
from environment.bitcoin_env import BitcoinTradingEnv
from models.dqn_agent import DQNAgent


def make_dummy_df(n: int = 10) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=n, freq="H")
    prices = np.linspace(100.0, 110.0, n)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": np.ones(n) * 10,
    })
    return df


def test_trainer_persists_episode_and_trades_best_effort():
    df = make_dummy_df(12)
    env = BitcoinTradingEnv(df, lookback_window=2, transaction_cost=0.0)
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=[32],
        use_per=False,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay=0.99,
        target_update_freq=10,
        batch_size=8,
    )

    mock_supabase = mock.MagicMock()
    mock_supabase.save_episode.return_value = {"success": True, "episode": {"id": 777}}
    mock_supabase.save_trades_batch.return_value = [{"success": True, "trade": {"id": 1}}]

    trainer = Trainer(
        agent=agent,
        env=env,
        eval_env=env,
        total_episodes=1,
        eval_frequency=100,
        checkpoint_frequency=1000,
        supabase_client=mock_supabase,
        agent_id="DQN-v2.1",
    )

    history = trainer.train()
    assert "episodes" in history

    # Verify Supabase methods were called
    assert mock_supabase.save_episode.called
    assert mock_supabase.save_trades_batch.called