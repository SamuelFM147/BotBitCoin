"""
Backtesting system for evaluating trading strategies
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime

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
            
            # Record portfolio value and position
            self.portfolio_values.append(info['total_value'])
            self.positions.append(info['btc_held'])
            
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
        
        return {
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
