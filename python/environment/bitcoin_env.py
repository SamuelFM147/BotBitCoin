"""
Bitcoin Trading Environment for Reinforcement Learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging
from numpy.random import default_rng, Generator
from dataclasses import dataclass
from utils.market_costs import SlippageConfig, compute_slippage_bps, apply_buy, apply_sell
from utils.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BitcoinTradingEnv(gym.Env):
    """
    Custom Trading Environment for Bitcoin
    Follows OpenAI Gym interface
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
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
                 fee_jitter_pct: float = 0.0):
        """
        Args:
            df: DataFrame with market data and features
            initial_balance: Starting cash balance
            lookback_window: Number of past observations
            transaction_cost: Trading fee as decimal (0.001 = 0.1%)
            max_position_size: Maximum position as fraction of balance
            reward_scaling: Scaling factor for rewards
        """
        super(BitcoinTradingEnv, self).__init__()
        
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
        
        # Get feature columns (exclude OHLCV and timestamp)
        self.feature_cols = [col for col in df.columns 
                            if col not in ['timestamp', 'date']]
        self.n_features = len(self.feature_cols)
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [lookback_window x n_features] + portfolio state
        obs_shape = (lookback_window * self.n_features + 3,)  # +3 for balance, position, price
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        # Último índice válido da série. O episódio começa em lookback_window
        # e termina quando current_step alcança o último índice.
        # Antes: usava (len(df) - lookback_window - 1), que representa a
        # quantidade de passos possíveis, mas era comparada diretamente com
        # current_step, resultando em episódios de 1 passo em datasets curtos.
        self.max_steps = len(df) - 1
        
        # Portfolio state
        self.balance = initial_balance
        self.btc_held = 0.0
        self.total_profit = 0.0
        self.trades = []
        self.rng: Generator | None = None
        self.risk_manager = RiskManager(max_position_size=max_position_size)
        self.entry_price: float | None = None
        self.prev_drawdown: float = 0.0
        self.prev_btc_held: float = 0.0
        
        logger.info(f"Environment initialized with {self.max_steps} steps")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.total_profit = 0.0
        self.trades = []
        self.rng = default_rng(seed if seed is not None else 42)
        self.entry_price = None
        self.prev_drawdown = 0.0
        self.prev_btc_held = 0.0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation
        Returns flattened array of [historical_features, portfolio_state]
        """
        # Get historical window
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        historical_data = self.df[self.feature_cols].iloc[start_idx:end_idx].values
        historical_flat = historical_data.flatten()
        
        # Current price
        current_price = self.df['close'].iloc[self.current_step]
        
        # Portfolio state
        total_value = self.balance + (self.btc_held * current_price)
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.btc_held * current_price / self.initial_balance,  # Position value normalized
            current_price / self.df['close'].iloc[self.lookback_window]  # Price relative to start
        ])
        
        # Combine
        observation = np.concatenate([historical_flat, portfolio_state])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, action: int, previous_value: float, delta_position: float) -> float:
        """
        Calculate reward for the action taken
        """
        # Evita estouro de índice quando current_step avança além do último índice
        safe_idx = min(self.current_step, len(self.df) - 1)
        current_price = self.df['close'].iloc[safe_idx]
        current_value = self.balance + (self.btc_held * current_price)
        
        # Portfolio value change
        value_change = (current_value - previous_value) / self.initial_balance
        
        # Fees/slippage already applied directly in balance and holdings.
        cost_penalty = 0.0
        if self.reward_include_fee_penalty and action != 0 and self.trades:
            lt = self.trades[-1]
            fee = float(lt.get('fee', 0.0) or 0.0)
            amt = float(lt.get('amount', 0.0) or 0.0)
            px = float(lt.get('price', current_price) or current_price)
            slip_bps = float(lt.get('slippage_bps', 0.0) or 0.0)
            slip_cost = (px * amt) * (abs(slip_bps) / 10000.0)
            cost_penalty = (fee + slip_cost) / max(self.initial_balance, 1e-12)
        
        # Risk-adjusted return (penalize excessive exposure)
        position_ratio = (self.btc_held * current_price) / current_value if current_value > 0 else 0
        risk_penalty = 0.0
        if position_ratio > self.max_position_size:
            risk_penalty = (position_ratio - self.max_position_size) * 0.1
        if self.lambda_inv > 0.0:
            risk_penalty += self.lambda_inv * abs(position_ratio)

        dd_prev = self.prev_drawdown
        dd_cur = self.risk_manager.update_drawdown(current_value)
        dd_delta = max(0.0, dd_cur - dd_prev)
        dd_penalty = self.lambda_dd * dd_delta

        turnover_penalty = self.lambda_turn * abs(delta_position)

        prices = self.df['close'].values
        start_idx = max(self.current_step - self.vol_window, 1)
        end_idx = self.current_step
        rets = np.diff(prices[start_idx-1:end_idx]) / prices[start_idx-1:end_idx-1] if end_idx - start_idx > 1 else np.array([0.0])
        sigma = float(np.std(rets))
        sigma_target = max(sigma, self.sigma_floor)
        
        # Combined reward
        base = (value_change - cost_penalty - risk_penalty - dd_penalty - turnover_penalty)
        reward = (base / sigma_target) * self.reward_scaling
        if self.reward_clip_abs is not None:
            reward = float(np.clip(reward, -abs(self.reward_clip_abs), abs(self.reward_clip_abs)))
        self.prev_drawdown = dd_cur
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 0 = Hold, 1 = Buy, 2 = Sell
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.df['close'].iloc[self.current_step]
        previous_value = self.balance + (self.btc_held * current_price)
        self.prev_btc_held = self.btc_held
        
        # Execute action with slippage & fees; enforce risk constraints
        if action == 1:  # Buy
            # Calculate how much we can buy (respect max position size)
            max_buy = (self.balance * self.max_position_size) / current_price
            buy_amount = max_buy * 0.5  # Buy 50% of maximum allowed

            # Risk check for position sizing
            total_value = self.balance + (self.btc_held * current_price)
            if not self.risk_manager.check_position_size(buy_amount * current_price, total_value):
                buy_amount = 0.0

            if buy_amount > 0 and self.balance > buy_amount * current_price:
                slip_bps = compute_slippage_bps(
                    self.df['close'].values,
                    self.current_step,
                    buy_amount,
                    'buy',
                    self.rng,
                    self.slippage_config,
                )
                eff_fee_rate = self.transaction_cost * (1.0 + (self.rng.normal(0.0, self.fee_jitter_pct) if self.fee_jitter_pct > 0 else 0.0))
                executed_price, cost, fee = apply_buy(current_price, buy_amount, eff_fee_rate, slip_bps)
                if self.balance >= cost:
                    self.balance -= cost
                    self.btc_held += buy_amount
                    self.entry_price = executed_price if self.entry_price is None else self.entry_price

                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'price': current_price,
                        'executed_price': executed_price,
                        'slippage_bps': slip_bps,
                        'amount': buy_amount,
                        'fee': fee,
                        'cost': cost,
                    })

        elif action == 2:  # Sell
            # Sell 50% of holdings
            sell_amount = self.btc_held * 0.5

            if sell_amount > 0:
                slip_bps = compute_slippage_bps(
                    self.df['close'].values,
                    self.current_step,
                    sell_amount,
                    'sell',
                    self.rng,
                    self.slippage_config,
                )
                eff_fee_rate = self.transaction_cost * (1.0 + (self.rng.normal(0.0, self.fee_jitter_pct) if self.fee_jitter_pct > 0 else 0.0))
                executed_price, revenue, fee = apply_sell(current_price, sell_amount, eff_fee_rate, slip_bps)
                self.balance += revenue
                self.btc_held -= sell_amount

                profit_loss = 0.0
                if self.entry_price is not None:
                    # PnL realized for the sold amount
                    profit_loss = (executed_price - self.entry_price) * sell_amount - fee
                    # Reset entry price if we fully exit
                    if self.btc_held <= 1e-12:
                        self.entry_price = None

                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'executed_price': executed_price,
                    'slippage_bps': slip_bps,
                    'amount': sell_amount,
                    'fee': fee,
                    'revenue': revenue,
                    'profit_loss': profit_loss,
                })

        # Risk-based auto close (stop loss / take profit) if holding
        if self.btc_held > 0 and self.entry_price is not None:
            check = self.risk_manager.should_close_position(self.entry_price, current_price, position_type='long')
            if check['should_close']:
                sell_amount = self.btc_held
                slip_bps = compute_slippage_bps(
                    self.df['close'].values,
                    self.current_step,
                    sell_amount,
                    'sell',
                    self.rng,
                    self.slippage_config,
                )
                eff_fee_rate = self.transaction_cost * (1.0 + (self.rng.normal(0.0, self.fee_jitter_pct) if self.fee_jitter_pct > 0 else 0.0))
                executed_price, revenue, fee = apply_sell(current_price, sell_amount, eff_fee_rate, slip_bps)
                self.balance += revenue
                self.btc_held = 0.0
                profit_loss = (executed_price - self.entry_price) * sell_amount - fee
                self.entry_price = None
                self.trades.append({
                    'step': self.current_step,
                    'action': 'auto_close',
                    'reason': check['reason'],
                    'price': current_price,
                    'executed_price': executed_price,
                    'slippage_bps': slip_bps,
                    'amount': sell_amount,
                    'fee': fee,
                    'revenue': revenue,
                    'profit_loss': profit_loss,
                })
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        delta_position = self.btc_held - self.prev_btc_held
        reward = self._calculate_reward(action, previous_value, delta_position)
        
        # Check if episode is done (atingiu último índice disponível)
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get new observation
        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape)
        
        # Info dict
        current_value = self.balance + (self.btc_held * current_price)
        info_cost_t = 0.0
        if self.reward_include_fee_penalty and action != 0 and self.trades:
            lt = self.trades[-1]
            fee_i = float(lt.get('fee', 0.0) or 0.0)
            amt_i = float(lt.get('amount', 0.0) or 0.0)
            px_i = float(lt.get('price', current_price) or current_price)
            slip_bps_i = float(lt.get('slippage_bps', 0.0) or 0.0)
            slip_cost_i = (px_i * amt_i) * (abs(slip_bps_i) / 10000.0)
            info_cost_t = (fee_i + slip_cost_i) / max(self.initial_balance, 1e-12)

        info = {
            'total_value': current_value,
            'profit': current_value - self.initial_balance,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'n_trades': len(self.trades),
            'last_trade': self.trades[-1] if self.trades else None,
            'sigma_target': sigma_target if 'sigma_target' in locals() else None,
            'drawdown': self.prev_drawdown,
            'dd_delta': dd_delta if 'dd_delta' in locals() else None,
            'turnover': abs(delta_position),
            'cost_t': info_cost_t,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment"""
        safe_idx = min(self.current_step, len(self.df) - 1)
        current_price = self.df['close'].iloc[safe_idx]
        current_value = self.balance + (self.btc_held * current_price)
        profit = current_value - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        print(f"\n{'='*50}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"BTC Held: {self.btc_held:.6f} BTC")
        print(f"Position Value: ${self.btc_held * current_price:.2f}")
        print(f"Total Value: ${current_value:.2f}")
        print(f"Profit: ${profit:.2f} ({profit_pct:.2f}%)")
        print(f"Trades: {len(self.trades)}")
        print(f"{'='*50}\n")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of portfolio performance"""
        current_price = self.df['close'].iloc[min(self.current_step, len(self.df) - 1)]
        final_value = self.balance + (self.btc_held * current_price)
        
        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_profit': final_value - self.initial_balance,
            'return_pct': ((final_value / self.initial_balance) - 1) * 100,
            'total_trades': len(self.trades),
            'final_balance': self.balance,
            'final_btc_held': self.btc_held
        }


if __name__ == "__main__":
    # Example usage
    # df = pd.read_csv('data/bitcoin_features.csv')
    # env = BitcoinTradingEnv(df)
    # obs, info = env.reset()
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    pass
