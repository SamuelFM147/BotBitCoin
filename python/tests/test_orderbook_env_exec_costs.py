import numpy as np
import pandas as pd

from utils.market_costs import SlippageConfig, compute_slippage_bps, apply_buy, apply_sell
from environment.orderbook_env import OrderBookTradingEnv


def make_df(prices):
    n = len(prices)
    return pd.DataFrame({
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.full(n, 1000.0, dtype=np.float32),
    })


def test_deterministic_slippage_and_fees_buy_sell():
    prices = np.array([10000, 10050, 10100, 10150, 10200], dtype=np.float32)
    df = make_df(prices)

    cfg = SlippageConfig(mu_bps=2.0, sigma_bps=0.0, scale_with_volatility=False)
    env = OrderBookTradingEnv(
        df=df,
        lookback_window=1,
        transaction_cost=0.001,
        slippage_config=cfg,
        fee_jitter_pct=0.0,
    )

    obs, _ = env.reset(seed=123)

    # BUY
    o, r, term, trunc, info = env.step(1)
    buy_trade = env.trades[-1]
    trade_step = buy_trade['step']
    amount = float(buy_trade['amount'])
    # usar mid/ask no passo do trade
    mid = env.df['_ob_mid'].values
    current_ask = float(env.df['_ob_ask'].iloc[trade_step])
    slip_bps = compute_slippage_bps(mid, trade_step, amount, 'buy', env.rng, cfg)
    exec_price, cost, fee = apply_buy(current_ask, amount, env.transaction_cost, slip_bps)

    assert np.isclose(buy_trade['executed_price'], exec_price, rtol=1e-4, atol=1e-6)
    assert np.isclose(buy_trade['cost'], cost, rtol=1e-4, atol=1e-6)
    assert np.isclose(buy_trade['fee'], fee, rtol=1e-4, atol=1e-6)

    # SELL
    o, r, term, trunc, info = env.step(2)
    sell_trade = env.trades[-1]
    trade_step = sell_trade['step']
    amount = float(sell_trade['amount'])
    current_bid = float(env.df['_ob_bid'].iloc[trade_step])
    slip_bps = compute_slippage_bps(mid, trade_step, amount, 'sell', env.rng, cfg)
    exec_price, revenue, fee = apply_sell(current_bid, amount, env.transaction_cost, slip_bps)

    assert np.isclose(sell_trade['executed_price'], exec_price, rtol=1e-4, atol=1e-6)
    assert np.isclose(sell_trade['revenue'], revenue, rtol=1e-4, atol=1e-6)
    assert np.isclose(sell_trade['fee'], fee, rtol=1e-4, atol=1e-6)

