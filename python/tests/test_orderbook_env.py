import pandas as pd
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from environment.orderbook_env import OrderBookTradingEnv


def test_orderbook_env_basic_step():
    pre = DataPreprocessor(scaling_method='standard')
    df = pre.load_data('python/data/bitcoin_historical.csv')
    df = pre.clean_data(df)
    eng = FeatureEngineer()
    df_feat = eng.engineer_features(df)
    df_norm = pre.normalize_data(df_feat, fit=True)
    env = OrderBookTradingEnv(df_norm, lookback_window=20)
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    done = False
    for _ in range(5):
        o, r, terminated, truncated, inf = env.step(env.action_space.sample())
        assert isinstance(r, float)
        assert o.shape[0] == env.observation_space.shape[0] or terminated
        done = terminated or truncated
        if done:
            break

