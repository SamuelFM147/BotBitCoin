"""
Environment factory
"""
from typing import Optional
import pandas as pd
from gymnasium.vector import SyncVectorEnv
from config.config import EnvironmentConfig
from environment.bitcoin_env import BitcoinTradingEnv
from environment.orderbook_env import OrderBookTradingEnv


def create_env(df: pd.DataFrame, cfg: EnvironmentConfig) -> object:
    if cfg.env_id == "orderbook_discrete":
        return OrderBookTradingEnv(
            df,
            initial_balance=cfg.initial_balance,
            lookback_window=cfg.lookback_window,
            transaction_cost=cfg.transaction_cost,
            max_position_size=cfg.max_position_size,
            reward_scaling=cfg.reward_scaling,
            reward_include_fee_penalty=cfg.reward_include_fee_penalty,
            vol_window=cfg.vol_window,
            sigma_floor=cfg.sigma_floor,
            lambda_dd=cfg.lambda_dd,
            lambda_inv=cfg.lambda_inv,
            lambda_turn=cfg.lambda_turn,
            reward_clip_abs=cfg.reward_clip_abs,
            fee_jitter_pct=cfg.fee_jitter_pct,
            n_levels=cfg.orderbook_levels,
            synthetic_orderbook=cfg.use_orderbook_synthetic,
        )
    return BitcoinTradingEnv(
        df,
        initial_balance=cfg.initial_balance,
        lookback_window=cfg.lookback_window,
        transaction_cost=cfg.transaction_cost,
        max_position_size=cfg.max_position_size,
        reward_scaling=cfg.reward_scaling,
        reward_include_fee_penalty=cfg.reward_include_fee_penalty,
        vol_window=cfg.vol_window,
        sigma_floor=cfg.sigma_floor,
        lambda_dd=cfg.lambda_dd,
        lambda_inv=cfg.lambda_inv,
        lambda_turn=cfg.lambda_turn,
        reward_clip_abs=cfg.reward_clip_abs,
        fee_jitter_pct=cfg.fee_jitter_pct,
    )


def create_vec_env(df: pd.DataFrame, cfg: EnvironmentConfig, num_envs: int) -> object:
    def make_thunk():
        if cfg.env_id == "orderbook_discrete":
            return OrderBookTradingEnv(
                df,
                initial_balance=cfg.initial_balance,
                lookback_window=cfg.lookback_window,
                transaction_cost=cfg.transaction_cost,
                max_position_size=cfg.max_position_size,
                reward_scaling=cfg.reward_scaling,
                reward_include_fee_penalty=cfg.reward_include_fee_penalty,
                vol_window=cfg.vol_window,
                sigma_floor=cfg.sigma_floor,
                lambda_dd=cfg.lambda_dd,
                lambda_inv=cfg.lambda_inv,
                lambda_turn=cfg.lambda_turn,
                reward_clip_abs=cfg.reward_clip_abs,
                fee_jitter_pct=cfg.fee_jitter_pct,
                n_levels=cfg.orderbook_levels,
                synthetic_orderbook=cfg.use_orderbook_synthetic,
            )
        return BitcoinTradingEnv(
            df,
            initial_balance=cfg.initial_balance,
            lookback_window=cfg.lookback_window,
            transaction_cost=cfg.transaction_cost,
            max_position_size=cfg.max_position_size,
            reward_scaling=cfg.reward_scaling,
            reward_include_fee_penalty=cfg.reward_include_fee_penalty,
            vol_window=cfg.vol_window,
            sigma_floor=cfg.sigma_floor,
            lambda_dd=cfg.lambda_dd,
            lambda_inv=cfg.lambda_inv,
            lambda_turn=cfg.lambda_turn,
            reward_clip_abs=cfg.reward_clip_abs,
            fee_jitter_pct=cfg.fee_jitter_pct,
        )

    return SyncVectorEnv([make_thunk for _ in range(int(num_envs))])

