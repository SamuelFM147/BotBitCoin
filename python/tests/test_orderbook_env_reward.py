import numpy as np
import pandas as pd

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


def test_volatility_scaling_magnitude():
    # baixa volatilidade: quase constante
    prices_low = np.full(120, 100.0, dtype=np.float32)
    # alta volatilidade: serrilhado
    prices_high = 100.0 + np.sin(np.linspace(0, 20, 120)).astype(np.float32) * 5.0

    df_low = make_df(prices_low)
    df_high = make_df(prices_high)

    lookback = 20
    sigma_floor = 1e-3

    env_low = OrderBookTradingEnv(df_low, lookback_window=lookback, sigma_floor=sigma_floor, fee_jitter_pct=0.0)
    env_high = OrderBookTradingEnv(df_high, lookback_window=lookback, sigma_floor=sigma_floor, fee_jitter_pct=0.0)

    obs, _ = env_low.reset(seed=123)
    o, r_low, t, tr, inf = env_low.step(1)  # compra para gerar variação

    obs, _ = env_high.reset(seed=123)
    o, r_high, t, tr, inf = env_high.step(1)

    # Com sigma_floor, baixa vol deve produzir magnitude maior que alta vol
    assert abs(r_low) > abs(r_high)


def test_penalties_lambda_inv_and_turnover():
    prices = 100.0 + 0.1 * np.arange(120, dtype=np.float32)
    df = make_df(prices)
    lookback = 20

    # Sem penalização
    env_base = OrderBookTradingEnv(df, lookback_window=lookback, lambda_inv=0.0, lambda_turn=0.0, fee_jitter_pct=0.0)
    obs, _ = env_base.reset(seed=123)
    o, r_base, t, tr, inf = env_base.step(1)  # delta_position > 0

    # Com penalização de inversão (exposição)
    env_inv = OrderBookTradingEnv(df, lookback_window=lookback, lambda_inv=0.5, lambda_turn=0.0, fee_jitter_pct=0.0)
    obs, _ = env_inv.reset(seed=123)
    o, r_inv, t, tr, inf = env_inv.step(1)
    assert r_inv < r_base

    # Com penalização de turnover
    env_turn = OrderBookTradingEnv(df, lookback_window=lookback, lambda_inv=0.0, lambda_turn=0.5, fee_jitter_pct=0.0)
    obs, _ = env_turn.reset(seed=123)
    o, r_turn, t, tr, inf = env_turn.step(1)
    assert r_turn < r_base


def test_drawdown_penalty_lambda_dd():
    # Preços caem após a compra para gerar piora no drawdown
    prices = np.concatenate([
        np.full(30, 100.0, dtype=np.float32),
        np.full(10, 98.0, dtype=np.float32),
        np.full(10, 97.0, dtype=np.float32),
    ])
    df = make_df(prices)
    lookback = 20

    # Ambiente sem penalização de drawdown
    env0 = OrderBookTradingEnv(df, lookback_window=lookback, lambda_dd=0.0, fee_jitter_pct=0.0)
    obs, _ = env0.reset(seed=123)
    # step 1: buy
    o, r1_0, t, tr, inf = env0.step(1)
    # step 2: hold em preço mais baixo → drawdown piora
    o, r2_0, t, tr, inf = env0.step(0)

    # Ambiente com penalização de drawdown
    env1 = OrderBookTradingEnv(df, lookback_window=lookback, lambda_dd=0.5, fee_jitter_pct=0.0)
    obs, _ = env1.reset(seed=123)
    o, r1_1, t, tr, inf = env1.step(1)
    o, r2_1, t, tr, inf = env1.step(0)

    # Penalização deve reduzir recompensa no segundo passo (quando dd_delta > 0)
    assert r2_1 < r2_0

