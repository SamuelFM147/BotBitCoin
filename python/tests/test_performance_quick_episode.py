import numpy as np
import pandas as pd

from config.config import Config
from environment.env_factory import create_env
from models.agent_factory import create_agent
from training.trainer import Trainer

def make_df(n: int = 120):
    prices = 100.0 + 0.1 * np.arange(n, dtype=np.float32)
    return pd.DataFrame({
        'close': prices,
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'volume': np.ones(n)
    })

def test_single_episode_duration_under_threshold():
    cfg = Config("config_fast.yaml")
    df = make_df(200)
    env = create_env(df, cfg.environment)
    agent, _ = create_agent("dqn", env, cfg)
    trainer = Trainer(
        agent=agent,
        env=env,
        eval_env=None,
        total_episodes=1,
        eval_frequency=1,
        checkpoint_frequency=1000,
        early_stopping_patience=1000,
        min_episodes_before_stopping=1,
        log_dir="logs",
        checkpoint_dir="checkpoints",
        supabase_client=None,
        agent_id="TEST-DQN",
        max_steps_per_episode=50,
        exploration_episodes=0,
        drawdown_max=None,
    )
    trainer._episode_idx = 0
    stats = trainer.train_episode()
    assert stats["steps"] > 0
    assert stats["duration_seconds"] < 5.0
