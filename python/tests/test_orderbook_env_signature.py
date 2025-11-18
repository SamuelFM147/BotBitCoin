import numpy as np
import pandas as pd

from environment.orderbook_env import OrderBookTradingEnv


def make_simple_df(n: int = 100, start: float = 100.0, step: float = 0.1) -> pd.DataFrame:
    prices = start + step * np.arange(n, dtype=np.float32)
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.full(n, 1000.0, dtype=np.float32),
    })
    return df


def test_gymnasium_signatures_and_spaces():
    lookback = 20
    df = make_simple_df(n=200)
    env = OrderBookTradingEnv(df=df, lookback_window=lookback, fee_jitter_pct=0.0)

    obs, info = env.reset(seed=123)
    assert isinstance(info, dict)
    assert isinstance(obs, np.ndarray)

    # step signature: (obs, reward, terminated, truncated, info)
    out = env.step(0)
    assert len(out) == 5
    o, r, terminated, truncated, inf = out
    assert isinstance(o, np.ndarray)
    assert isinstance(r, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(inf, dict)

    # observation space and action space
    expected_obs_dim = lookback * 3 + 3
    assert env.observation_space.shape == (expected_obs_dim,)
    assert env.action_space.n == 3


def test_observation_shape_and_content():
    lookback = 15
    df = make_simple_df(n=120)
    env = OrderBookTradingEnv(df=df, lookback_window=lookback, fee_jitter_pct=0.0)

    obs, _ = env.reset(seed=123)
    assert obs.shape == (lookback * 3 + 3,)

    # Recalcular bid/ask/spread da janela para validar conteúdo do hist
    s = env.current_step - lookback
    e = env.current_step
    bid = env.df['_ob_bid'].iloc[s:e].values
    ask = env.df['_ob_ask'].iloc[s:e].values
    spread = (ask - bid) / np.maximum(env.df['_ob_mid'].iloc[s:e].values, 1e-12)
    hist_expected = np.stack([bid, ask, spread], axis=1).astype(np.float32).reshape(-1)

    # Portfolio: [balance_norm, position_norm, price_rel]
    current_mid = float(env.df['_ob_mid'].iloc[env.current_step])
    total_value = env.balance + (env.asset_held * current_mid)
    portfolio_expected = np.array([
        env.balance / env.initial_balance,
        (env.asset_held * current_mid) / env.initial_balance,
        current_mid / max(env.df['_ob_mid'].iloc[env.lookback_window], 1e-12),
    ], dtype=np.float32)

    # Validar conteúdo com tolerância
    np.testing.assert_allclose(obs[: lookback * 3], hist_expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(obs[-3:], portfolio_expected, rtol=1e-4, atol=1e-6)

