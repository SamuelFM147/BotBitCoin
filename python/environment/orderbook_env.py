"""
Order book based trading environment
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from numpy.random import default_rng, Generator
from utils.market_costs import SlippageConfig, compute_slippage_bps, apply_buy, apply_sell
from utils.risk_manager import RiskManager


class OrderBookTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        lookback_window: int = 50,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.3,
        reward_scaling: float = 1.0,
        slippage_config: SlippageConfig | None = None,
        reward_include_fee_penalty: bool = True,
        vol_window: int = 50,
        sigma_floor: float = 1e-6,
        lambda_dd: float = 0.1,
        lambda_inv: float = 0.0,
        lambda_turn: float = 0.0,
        reward_clip_abs: float | None = None,
        fee_jitter_pct: float = 0.0,
        n_levels: int = 5,
        synthetic_orderbook: bool = True,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.reward_scaling = reward_scaling
        self.slippage_config = slippage_config or SlippageConfig()
        self.reward_include_fee_penalty = reward_include_fee_penalty
        self.vol_window = vol_window
        self.sigma_floor = sigma_floor
        self.lambda_dd = lambda_dd
        self.lambda_inv = lambda_inv
        self.lambda_turn = lambda_turn
        self.reward_clip_abs = reward_clip_abs
        self.fee_jitter_pct = fee_jitter_pct
        self.n_levels = max(1, int(n_levels))
        self.synthetic_orderbook = synthetic_orderbook

        self._prepare_orderbook_features()

        self.action_space = spaces.Discrete(3)

        obs_dim = self.lookback_window * (3) + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.balance = initial_balance
        self.asset_held = 0.0
        self.total_profit = 0.0
        self.trades = []
        self.rng: Generator | None = None
        self.risk_manager = RiskManager(max_position_size=max_position_size)
        self.entry_price: float | None = None
        self.prev_drawdown: float = 0.0
        self.prev_asset_held: float = 0.0

    def _prepare_orderbook_features(self):
        close = self.df['close'].astype(float).values
        spread_proxy = None
        if 'hl_spread' in self.df.columns:
            spread_proxy = self.df['hl_spread'].astype(float).values
        else:
            spread_proxy = np.zeros_like(close)
        base_spread = np.clip(spread_proxy, 0.0, 0.02)
        bid = close * (1.0 - base_spread * 0.5)
        ask = close * (1.0 + base_spread * 0.5)
        mid = (bid + ask) / 2.0
        self.df['_ob_bid'] = bid
        self.df['_ob_ask'] = ask
        self.df['_ob_mid'] = mid

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.asset_held = 0.0
        self.total_profit = 0.0
        self.trades = []
        self.rng = default_rng(seed if seed is not None else 42)
        self.entry_price = None
        self.prev_drawdown = 0.0
        self.prev_asset_held = 0.0
        self.risk_manager.peak_value = self.initial_balance
        self.risk_manager.current_drawdown = 0.0
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        s = self.current_step - self.lookback_window
        e = self.current_step
        bid = self.df['_ob_bid'].iloc[s:e].values
        ask = self.df['_ob_ask'].iloc[s:e].values
        spread = (ask - bid) / np.maximum(self.df['_ob_mid'].iloc[s:e].values, 1e-12)
        hist = np.stack([bid, ask, spread], axis=1).astype(np.float32).reshape(-1)
        current_mid = float(self.df['_ob_mid'].iloc[self.current_step])
        total_value = self.balance + (self.asset_held * current_mid)
        portfolio = np.array([
            self.balance / self.initial_balance,
            (self.asset_held * current_mid) / self.initial_balance,
            current_mid / max(self.df['_ob_mid'].iloc[self.lookback_window], 1e-12),
        ], dtype=np.float32)
        return np.concatenate([hist, portfolio]).astype(np.float32)

    def _calculate_reward(self, action: int, previous_value: float, delta_position: float) -> float:
        idx = min(self.current_step, len(self.df) - 1)
        current_mid = float(self.df['_ob_mid'].iloc[idx])
        current_value = self.balance + (self.asset_held * current_mid)
        value_change = (current_value - previous_value) / max(self.initial_balance, 1e-12)
        cost_penalty = 0.0
        if self.reward_include_fee_penalty and action != 0 and self.trades:
            lt = self.trades[-1]
            fee = float(lt.get('fee', 0.0) or 0.0)
            amt = float(lt.get('amount', 0.0) or 0.0)
            px = float(lt.get('executed_price', current_mid) or current_mid)
            slip_bps = float(lt.get('slippage_bps', 0.0) or 0.0)
            slip_cost = (px * amt) * (abs(slip_bps) / 10000.0)
            cost_penalty = (fee + slip_cost) / max(self.initial_balance, 1e-12)
        position_ratio = (self.asset_held * current_mid) / current_value if current_value > 0 else 0.0
        risk_penalty = 0.0
        if position_ratio > self.max_position_size:
            risk_penalty = (position_ratio - self.max_position_size) * 0.1
        if self.lambda_inv > 0.0:
            risk_penalty += self.lambda_inv * abs(position_ratio)
        dd_prev = self.prev_drawdown
        dd_cur = self.risk_manager.update_drawdown(current_value)
        dd_delta = max(0.0, dd_cur - dd_prev)
        dd_penalty = self.lambda_dd * dd_cur
        turnover_penalty = self.lambda_turn * abs(delta_position)
        mids = self.df['_ob_mid'].values
        start_idx = max(self.current_step - self.vol_window, 1)
        end_idx = self.current_step
        rets = np.diff(mids[start_idx-1:end_idx]) / mids[start_idx-1:end_idx-1] if end_idx - start_idx > 1 else np.array([0.0])
        sigma = float(np.std(rets))
        sigma_target = max(sigma, self.sigma_floor)
        base = (value_change - cost_penalty - risk_penalty - dd_penalty - turnover_penalty)
        reward = (base / sigma_target) * self.reward_scaling
        if self.reward_clip_abs is not None:
            reward = float(np.clip(reward, -abs(self.reward_clip_abs), abs(self.reward_clip_abs)))
        self.prev_drawdown = dd_cur
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        current_bid = float(self.df['_ob_bid'].iloc[self.current_step])
        current_ask = float(self.df['_ob_ask'].iloc[self.current_step])
        current_mid = float(self.df['_ob_mid'].iloc[self.current_step])
        previous_value = self.balance + (self.asset_held * current_mid)
        self.prev_asset_held = self.asset_held
        if action == 1:
            max_buy = (self.balance * self.max_position_size) / max(current_ask, 1e-12)
            buy_amount = max_buy * 0.5
            total_value = self.balance + (self.asset_held * current_mid)
            if not self.risk_manager.check_position_size(buy_amount * current_ask, total_value):
                buy_amount = 0.0
            if buy_amount > 0 and self.balance > buy_amount * current_ask:
                slip_bps = compute_slippage_bps(self.df['_ob_mid'].values, self.current_step, buy_amount, 'buy', self.rng, self.slippage_config)
                eff_fee_rate = self.transaction_cost * (1.0 + (self.rng.normal(0.0, self.fee_jitter_pct) if self.fee_jitter_pct > 0 else 0.0))
                executed_price, cost, fee = apply_buy(current_ask, buy_amount, eff_fee_rate, slip_bps)
                if self.balance >= cost:
                    self.balance -= cost
                    self.asset_held += buy_amount
                    self.entry_price = executed_price if self.entry_price is None else self.entry_price
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'executed_price': executed_price,
                        'slippage_bps': slip_bps,
                        'amount': buy_amount,
                        'fee': fee,
                        'cost': cost,
                    })
        elif action == 2:
            sell_amount = self.asset_held * 0.5
            if sell_amount > 0:
                slip_bps = compute_slippage_bps(self.df['_ob_mid'].values, self.current_step, sell_amount, 'sell', self.rng, self.slippage_config)
                eff_fee_rate = self.transaction_cost * (1.0 + (self.rng.normal(0.0, self.fee_jitter_pct) if self.fee_jitter_pct > 0 else 0.0))
                executed_price, revenue, fee = apply_sell(current_bid, sell_amount, eff_fee_rate, slip_bps)
                self.balance += revenue
                self.asset_held -= sell_amount
                profit_loss = 0.0
                if self.entry_price is not None:
                    profit_loss = (executed_price - self.entry_price) * sell_amount - fee
                    if self.asset_held <= 1e-12:
                        self.entry_price = None
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'executed_price': executed_price,
                    'slippage_bps': slip_bps,
                    'amount': sell_amount,
                    'fee': fee,
                    'revenue': revenue,
                    'profit_loss': profit_loss,
                })
        if self.asset_held > 0 and self.entry_price is not None:
            check = self.risk_manager.should_close_position(self.entry_price, current_mid, position_type='long')
            if check['should_close']:
                sell_amount = self.asset_held
                slip_bps = compute_slippage_bps(self.df['_ob_mid'].values, self.current_step, sell_amount, 'sell', self.rng, self.slippage_config)
                eff_fee_rate = self.transaction_cost * (1.0 + (self.rng.normal(0.0, self.fee_jitter_pct) if self.fee_jitter_pct > 0 else 0.0))
                executed_price, revenue, fee = apply_sell(current_bid, sell_amount, eff_fee_rate, slip_bps)
                self.balance += revenue
                self.asset_held = 0.0
                profit_loss = (executed_price - self.entry_price) * sell_amount - fee
                self.entry_price = None
                self.trades.append({
                    'step': self.current_step,
                    'action': 'auto_close',
                    'reason': check['reason'],
                    'executed_price': executed_price,
                    'slippage_bps': slip_bps,
                    'amount': sell_amount,
                    'fee': fee,
                    'revenue': revenue,
                    'profit_loss': profit_loss,
                })
        self.current_step += 1
        delta_position = self.asset_held - self.prev_asset_held
        reward = self._calculate_reward(action, previous_value, delta_position)
        terminated = self.current_step >= self.max_steps
        truncated = False
        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        current_value = self.balance + (self.asset_held * current_mid)
        info_cost_t = 0.0
        if self.reward_include_fee_penalty and action != 0 and self.trades:
            lt = self.trades[-1]
            fee_i = float(lt.get('fee', 0.0) or 0.0)
            amt_i = float(lt.get('amount', 0.0) or 0.0)
            px_i = float(lt.get('executed_price', current_mid) or current_mid)
            slip_bps_i = float(lt.get('slippage_bps', 0.0) or 0.0)
            slip_cost_i = (px_i * amt_i) * (abs(slip_bps_i) / 10000.0)
            info_cost_t = (fee_i + slip_cost_i) / max(self.initial_balance, 1e-12)
        info = {
            'total_value': current_value,
            'profit': current_value - self.initial_balance,
            'balance': self.balance,
            'asset_held': self.asset_held,
            'n_trades': len(self.trades),
            'last_trade': self.trades[-1] if self.trades else None,
            'drawdown': self.prev_drawdown,
            'turnover': abs(delta_position),
            'cost_t': info_cost_t,
        }
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        idx = min(self.current_step, len(self.df) - 1)
        current_mid = float(self.df['_ob_mid'].iloc[idx])
        current_value = self.balance + (self.asset_held * current_mid)
        profit = current_value - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100.0
        print(f"Step: {self.current_step}/{self.max_steps} | Mid: {current_mid:.6f} | Balance: {self.balance:.2f} | Held: {self.asset_held:.6f} | Value: {current_value:.2f} | PnL: {profit:.2f} ({profit_pct:.2f}%)")
