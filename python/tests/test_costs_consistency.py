import numpy as np
import pandas as pd

from python.utils.market_costs import SlippageConfig, compute_slippage_bps, apply_buy, apply_sell
from python.environment.bitcoin_env import BitcoinTradingEnv


def make_df(prices):
    return pd.DataFrame({
        'close': prices,
        # minimal feature set: use close itself
    })


def test_env_trade_costs_match_utils():
    prices = np.array([10000, 10100, 10200, 10300, 10400], dtype=np.float32)
    df = make_df(prices)
    cfg = SlippageConfig(mu_bps=2.0, sigma_bps=0.0, scale_with_volatility=False)  # deterministic
    env = BitcoinTradingEnv(df=df, lookback_window=1, transaction_cost=0.001, slippage_config=cfg)

    obs, _ = env.reset(seed=123)

    # Step 1: Buy
    current_price = df['close'].iloc[env.current_step]
    action_buy = 1
    obs, reward, terminated, truncated, info = env.step(action_buy)
    buy_trade = env.trades[-1]

    amount = buy_trade['amount']
    slip_bps = compute_slippage_bps(df['close'].values, env.current_step - 1, amount, 'buy', env.rng, cfg)
    exec_price, cost, fee = apply_buy(current_price, amount, env.transaction_cost, slip_bps)

    assert np.isclose(buy_trade['executed_price'], exec_price)
    assert np.isclose(buy_trade['cost'], cost)
    assert np.isclose(buy_trade['fee'], fee)

    # Step 2: Sell
    current_price = df['close'].iloc[env.current_step]
    action_sell = 2
    obs, reward, terminated, truncated, info = env.step(action_sell)
    sell_trade = env.trades[-1]

    amount = sell_trade['amount']
    slip_bps = compute_slippage_bps(df['close'].values, env.current_step - 1, amount, 'sell', env.rng, cfg)
    exec_price, revenue, fee = apply_sell(current_price, amount, env.transaction_cost, slip_bps)

    assert np.isclose(sell_trade['executed_price'], exec_price)
    assert np.isclose(sell_trade['revenue'], revenue)
    assert np.isclose(sell_trade['fee'], fee)