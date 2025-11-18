import numpy as np
import pandas as pd
from python.evaluation.backtester import Backtester
from python.environment.bitcoin_env import BitcoinTradingEnv


class RandomAgent:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    def select_action(self, state, training=False):
        import random
        return random.randrange(self.action_dim)


def test_backtester_metrics_keys():
    n = 300
    prices = np.linspace(30000, 30500, n) + np.random.randn(n) * 30
    df = pd.DataFrame({'close': prices, 'feat': np.random.randn(n)})
    env = BitcoinTradingEnv(df)
    agent = RandomAgent(env.action_space.n)
    bt = Backtester(initial_balance=10000.0, transaction_cost=0.001)
    res = bt.run_backtest(agent, env, verbose=False)
    metrics = res['metrics']
    for key in [
        'profit_factor', 'turnover', 'exposure',
        'avg_fee_per_trade', 'avg_slippage_bps',
        'alpha_vs_benchmark', 'beta_vs_benchmark', 'sharpe_stability',
    ]:
        assert key in metrics


def test_walk_forward_outputs():
    n = 400
    prices = np.linspace(30000, 31000, n) + np.random.randn(n) * 25
    df = pd.DataFrame({'close': prices, 'feat': np.random.randn(n)})
    env = BitcoinTradingEnv(df)
    agent = RandomAgent(env.action_space.n)
    bt = Backtester(initial_balance=10000.0, transaction_cost=0.001)
    wf = bt.walk_forward(agent, df, lookback_window=env.lookback_window, window_size=200, step_size=150, verbose=False)
    assert isinstance(wf, list) and len(wf) >= 1
    sample = wf[0]
    assert 'metrics' in sample and isinstance(sample['metrics'], dict)
