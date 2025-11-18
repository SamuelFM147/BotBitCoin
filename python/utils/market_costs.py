"""
Shared market cost and slippage utilities to ensure consistency
between environment and backtester.

Slippage expressed in basis points (bps): 1 bps = 0.01%
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class SlippageConfig:
    mu_bps: float = 2.0               # baseline slippage (bps)
    sigma_bps: float = 1.0            # randomness (bps)
    scale_with_volatility: bool = False
    vol_window: int = 20              # window for realized volatility
    size_sensitivity: float = 0.0     # bps per unit of (amount / 1 BTC)


def realized_vol_bps(prices: np.ndarray, idx: int, window: int) -> float:
    """Compute simple realized volatility in bps from returns over window."""
    start = max(0, idx - window)
    segment = prices[start:idx+1]
    if len(segment) < 2:
        return 0.0
    rets = np.diff(segment) / segment[:-1]
    vol = np.std(rets) * 1e4  # convert to bps
    return float(vol)


def compute_slippage_bps(
    prices: np.ndarray,
    idx: int,
    amount: float,
    side: str,
    rng: np.random.Generator,
    config: Optional[SlippageConfig] = None,
) -> float:
    """
    Compute slippage in basis points for a trade.

    Args:
        prices: array of close prices
        idx: current index in prices
        amount: trade size in BTC (positive number)
        side: 'buy' or 'sell'
        rng: numpy RNG for reproducibility
        config: SlippageConfig
    Returns:
        slippage in bps (positive number); applied + for buy, - for sell
    """
    if config is None:
        config = SlippageConfig()

    base = config.mu_bps
    vol_bps = realized_vol_bps(prices, idx, config.vol_window) if config.scale_with_volatility else 0.0
    size_bps = config.size_sensitivity * max(0.0, amount)

    noise = rng.normal(loc=0.0, scale=config.sigma_bps)
    slip_bps = max(0.0, base + 1.0 * vol_bps + size_bps + noise)

    return float(slip_bps)


def apply_buy(price: float, amount: float, fee_rate: float, slippage_bps: float) -> Tuple[float, float, float]:
    """
    Apply slippage and fee to a buy.
    Returns: executed_price, cost, fee
    """
    executed_price = price * (1.0 + slippage_bps / 1e4)
    gross_cost = amount * executed_price
    fee = gross_cost * fee_rate
    cost = gross_cost + fee
    return executed_price, cost, fee


def apply_sell(price: float, amount: float, fee_rate: float, slippage_bps: float) -> Tuple[float, float, float]:
    """
    Apply slippage and fee to a sell.
    Returns: executed_price, revenue, fee
    """
    executed_price = price * (1.0 - slippage_bps / 1e4)
    gross_rev = amount * executed_price
    fee = gross_rev * fee_rate
    revenue = gross_rev - fee
    return executed_price, revenue, fee
