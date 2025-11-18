import numpy as np
import pandas as pd

from config.config import EnvironmentConfig
from environment.env_factory import create_env
from environment.orderbook_env import OrderBookTradingEnv
from environment.bitcoin_env import BitcoinTradingEnv


def make_df(n: int = 100):
    prices = 100.0 + 0.1 * np.arange(n, dtype=np.float32)
    return pd.DataFrame({'close': prices, 'open': prices, 'high': prices * 1.001, 'low': prices * 0.999, 'volume': np.ones(n)})


def test_env_selection_and_param_propagation():
    df = make_df(120)
    cfg = EnvironmentConfig()

    # orderbook_discrete → OrderBookTradingEnv
    cfg.env_id = 'orderbook_discrete'
    cfg.orderbook_levels = 7
    cfg.use_orderbook_synthetic = False
    env_ob = create_env(df, cfg)
    assert isinstance(env_ob, OrderBookTradingEnv)
    assert env_ob.n_levels == 7
    assert env_ob.synthetic_orderbook is False

    # default (ohlcv_discrete) → BitcoinTradingEnv
    cfg.env_id = 'ohlcv_discrete'
    env_btc = create_env(df, cfg)
    assert isinstance(env_btc, BitcoinTradingEnv)

