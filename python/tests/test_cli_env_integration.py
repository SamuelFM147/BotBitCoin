import numpy as np
import pandas as pd

from config.config import Config
from environment.env_factory import create_env
from environment.orderbook_env import OrderBookTradingEnv
from environment.bitcoin_env import BitcoinTradingEnv


def make_df(n: int = 100):
    prices = 100.0 + 0.1 * np.arange(n, dtype=np.float32)
    return pd.DataFrame({'close': prices, 'open': prices, 'high': prices * 1.001, 'low': prices * 0.999, 'volume': np.ones(n)})


def test_cli_env_flag_applied_to_config_and_factory():
    df = make_df(120)

    # Simular args.env → Config.environment.env_id
    config = Config()
    assert config.environment.env_id == 'ohlcv_discrete'

    # "--env orderbook_discrete"
    config.environment.env_id = 'orderbook_discrete'
    env = create_env(df, config.environment)
    assert isinstance(env, OrderBookTradingEnv)

    # "--env ohlcv_discrete" (default)
    config.environment.env_id = 'ohlcv_discrete'
    env2 = create_env(df, config.environment)
    assert isinstance(env2, BitcoinTradingEnv)

