"""
Backtesting system for evaluating trading strategies
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime
from utils.risk_manager import RiskManager
from environment.bitcoin_env import BitcoinTradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine for trading strategies
    """
    
    def __init__(self,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001):
        """
        Args:
            initial_balance: Starting capital
            transaction_cost: Trading fee as decimal
        """
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Results storage
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        self.prices = []
        
    def run_backtest(self, 
                    agent,
                    env,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Run backtest using trained agent
        
        Args:
            agent: Trained RL agent
            env: Trading environment with test data
            verbose: If True, print progress
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")
        
        # Reset environment and tracking
        state, _ = env.reset()
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        
        done = False
        step = 0
        
        while not done:
            # Get action from agent
            action = agent.select_action(state, training=False)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record portfolio value, position and price
            self.portfolio_values.append(info['total_value'])
            self.positions.append(info['btc_held'])
            try:
                price_idx = max(env.current_step - 1, 0)
                self.prices.append(float(env.df['close'].iloc[price_idx]))
            except Exception:
                self.prices.append(np.nan)
            
            # Record trades
            if len(env.trades) > len(self.trades):
                self.trades.extend(env.trades[len(self.trades):])
            
            state = next_state
            step += 1
            
            if verbose and step % 100 == 0:
                logger.info(f"Step {step}: Value=${info['total_value']:.2f}, "
                          f"Profit=${info['profit']:.2f}")
        
        # Get final portfolio summary
        summary = env.get_portfolio_summary()
        
        # Calculate metrics
        metrics = self.calculate_metrics(summary)
        
        logger.info("Backtest completed!")
        logger.info(f"Final value: ${summary['final_value']:.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2f}%")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        
        return {
            'summary': summary,
            'metrics': metrics,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'positions': self.positions
        }
    
    def calculate_metrics(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            summary: Portfolio summary from environment
            
        Returns:
            Dictionary of metrics
        """
        # Convert portfolio values to returns
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        # Total return
        total_return = ((summary['final_value'] / summary['initial_balance']) - 1) * 100
        
        # Sharpe ratio (assuming 252 trading days per year, 0% risk-free rate)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns))
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        max_drawdown = self.calculate_max_drawdown(self.portfolio_values)
        
        # Win rate
        if len(self.trades) > 0:
            profitable_trades = sum(1 for trade in self.trades 
                                   if trade.get('profit_loss', 0) > 0)
            win_rate = (profitable_trades / len(self.trades)) * 100
        else:
            win_rate = 0.0
        
        # Volatility (annualized)
        if len(returns) > 0:
            volatility = np.std(returns) * np.sqrt(252) * 100
        else:
            volatility = 0.0
        
        # Calmar ratio (return / max_drawdown)
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0.0
        
        # Sortino ratio (downside deviation)
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_ratio = np.sqrt(252) * (np.mean(returns) / downside_std)
            else:
                sortino_ratio = float('inf')
        else:
            sortino_ratio = 0.0
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': len(self.trades),
            'avg_trade_profit': np.mean([t.get('profit_loss', 0) for t in self.trades]) if self.trades else 0
        }

        try:
            gross_profit = sum(float(t.get('profit_loss', 0) or 0) for t in self.trades if float(t.get('profit_loss', 0) or 0) > 0)
            gross_loss = -sum(float(t.get('profit_loss', 0) or 0) for t in self.trades if float(t.get('profit_loss', 0) or 0) < 0)
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
            turnover = sum(abs(float(t.get('amount', 0) or 0) * float(t.get('executed_price', t.get('price', 0)) or 0)) for t in self.trades) / self.initial_balance if self.initial_balance > 0 else 0.0
            metrics.update({'profit_factor': profit_factor, 'turnover': turnover})
        except Exception:
            pass

        # Risk metrics (VaR, CVaR, current drawdown, downside deviation)
        try:
            rm = RiskManager()
            risk_metrics = rm.get_risk_metrics(np.array(self.portfolio_values))
            metrics.update(risk_metrics)
        except Exception as e:
            logger.warning(f"Risk metrics calculation failed: {e}")

        try:
            if len(self.positions) > 0 and len(self.prices) == len(self.positions):
                expo = []
                for i in range(len(self.positions)):
                    val = self.portfolio_values[i]
                    pr = self.prices[i]
                    pos_val = abs(self.positions[i] * pr)
                    ratio = (pos_val / val) if val > 0 else 0.0
                    expo.append(ratio)
                exposure = float(np.mean(expo))
                metrics['exposure'] = exposure
        except Exception:
            pass

        try:
            fees = [float(t.get('fee', 0) or 0) for t in self.trades]
            slbps = [abs(float(t.get('slippage_bps', 0) or 0)) for t in self.trades]
            metrics['avg_fee_per_trade'] = float(np.mean(fees)) if fees else 0.0
            metrics['avg_slippage_bps'] = float(np.mean(slbps)) if slbps else 0.0
        except Exception:
            pass

        try:
            if len(returns) > 0 and len(self.prices) > 1:
                bench_ret = np.diff(np.array(self.prices)) / np.array(self.prices[:-1])
                m = min(len(bench_ret), len(returns))
                br = bench_ret[:m]
                sr = returns[:m]
                var_b = np.var(br)
                cov = np.cov(sr, br)[0, 1]
                beta = (cov / var_b) if var_b > 0 else 0.0
                alpha = float(np.mean(sr) - beta * np.mean(br))
                metrics['beta_vs_benchmark'] = beta
                metrics['alpha_vs_benchmark'] = alpha * np.sqrt(252)
        except Exception:
            pass

        try:
            if len(returns) > 0:
                block = max(10, int(len(returns) / 10))
                sharpe_blocks = []
                for i in range(0, len(returns), block):
                    seg = returns[i:i+block]
                    if len(seg) > 1 and np.std(seg) > 0:
                        sharpe_blocks.append(np.sqrt(252) * (np.mean(seg) / np.std(seg)))
                if sharpe_blocks:
                    metrics['sharpe_stability'] = float(np.std(sharpe_blocks))
        except Exception:
            pass

        return metrics
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            portfolio_values: List of portfolio values over time
            
        Returns:
            Maximum drawdown as percentage
        """
        if len(portfolio_values) == 0:
            return 0.0
        
        portfolio_values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max * 100
        max_drawdown = np.min(drawdowns)
        
        return abs(max_drawdown)
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """
        Get detailed trade analysis
        
        Returns:
            DataFrame with trade statistics
        """
        if len(self.trades) == 0:
            return pd.DataFrame()
        
        df_trades = pd.DataFrame(self.trades)
        
        # Add profit/loss if not present
        if 'profit_loss' not in df_trades.columns:
            df_trades['profit_loss'] = 0
        
        # Calculate cumulative profit
        df_trades['cumulative_profit'] = df_trades['profit_loss'].cumsum()
        
        return df_trades

    def walk_forward(self,
                     agent,
                     df: pd.DataFrame,
                     lookback_window: int,
                     window_size: int,
                     step_size: int,
                     verbose: bool = True) -> List[Dict[str, Any]]:
        results = []
        n = len(df)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            df_slice = df.iloc[start:end].reset_index(drop=True)
            env = BitcoinTradingEnv(
                df_slice,
                initial_balance=self.initial_balance,
                lookback_window=lookback_window,
                transaction_cost=self.transaction_cost,
                max_position_size=0.3,
            )
            _ = self.run_backtest(agent, env, verbose=False)
            metrics = self.calculate_metrics(env.get_portfolio_summary())
            results.append({
                'start': int(start),
                'end': int(end),
                'metrics': metrics,
            })
            if verbose:
                logger.info(f"WF [{start}:{end}] Sharpe={metrics.get('sharpe_ratio', 0):.3f} DD={metrics.get('max_drawdown', 0):.2f}% Return={metrics.get('total_return', 0):.2f}%")
            start += step_size
        return results
    
    def plot_results(self, save_path: str = None):
        """
        Plot backtest results
        
        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            
            # Portfolio value over time
            axes[0].plot(self.portfolio_values, label='Portfolio Value')
            axes[0].axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Balance')
            axes[0].set_title('Portfolio Value Over Time')
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('Value ($)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Returns
            if len(self.portfolio_values) > 1:
                returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1] * 100
                axes[1].plot(returns, label='Returns (%)', alpha=0.7)
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].set_title('Returns Over Time')
                axes[1].set_xlabel('Steps')
                axes[1].set_ylabel('Return (%)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            # Cumulative returns
            if len(self.portfolio_values) > 0:
                cum_returns = (np.array(self.portfolio_values) / self.initial_balance - 1) * 100
                axes[2].plot(cum_returns, label='Cumulative Return (%)', color='green')
                axes[2].axhline(y=0, color='r', linestyle='--')
                axes[2].set_title('Cumulative Return Over Time')
                axes[2].set_xlabel('Steps')
                axes[2].set_ylabel('Cumulative Return (%)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


if __name__ == "__main__":
    # Example usage
    # from models.dqn_agent import DQNAgent
    # from environment.bitcoin_env import BitcoinTradingEnv
    # import pandas as pd
    
    # # Load test data
    # df_test = pd.read_csv('data/bitcoin_test.csv')
    
    # # Create environment
    # env = BitcoinTradingEnv(df_test)
    
    # # Load trained agent
    # agent = DQNAgent(state_dim, action_dim)
    # agent.load('checkpoints/best_model.pth')
    
    # # Run backtest
    # backtester = Backtester()
    # results = backtester.run_backtest(agent, env)
    # backtester.plot_results('results/backtest_plot.png')
    pass
