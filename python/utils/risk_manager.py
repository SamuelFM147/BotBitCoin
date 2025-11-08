"""
Risk management utilities for trading
"""
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages risk for trading strategies
    """
    
    def __init__(self,
                 max_position_size: float = 0.3,
                 max_drawdown_limit: float = 0.20,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10):
        """
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_drawdown_limit: Maximum allowed drawdown before stopping
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.max_position_size = max_position_size
        self.max_drawdown_limit = max_drawdown_limit
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.peak_value = 0
        self.current_drawdown = 0
        
    def check_position_size(self, 
                           position_value: float, 
                           total_value: float) -> bool:
        """
        Check if position size is within limits
        
        Args:
            position_value: Current position value
            total_value: Total portfolio value
            
        Returns:
            True if within limits, False otherwise
        """
        if total_value == 0:
            return False
        
        position_ratio = position_value / total_value
        return position_ratio <= self.max_position_size
    
    def calculate_position_size(self, 
                               capital: float, 
                               risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on risk
        
        Args:
            capital: Available capital
            risk_per_trade: Risk per trade as fraction (default 2%)
            
        Returns:
            Position size
        """
        return capital * risk_per_trade
    
    def update_drawdown(self, current_value: float) -> float:
        """
        Update and return current drawdown
        
        Args:
            current_value: Current portfolio value
            
        Returns:
            Current drawdown as decimal
        """
        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0
        
        return self.current_drawdown
    
    def should_stop_trading(self, current_value: float) -> bool:
        """
        Check if trading should be stopped due to risk limits
        
        Args:
            current_value: Current portfolio value
            
        Returns:
            True if should stop, False otherwise
        """
        drawdown = self.update_drawdown(current_value)
        
        if drawdown >= self.max_drawdown_limit:
            logger.warning(f"Maximum drawdown limit reached: {drawdown:.2%}")
            return True
        
        return False
    
    def should_close_position(self,
                             entry_price: float,
                             current_price: float,
                             position_type: str = 'long') -> Dict[str, Any]:
        """
        Check if position should be closed based on stop loss/take profit
        
        Args:
            entry_price: Entry price
            current_price: Current price
            position_type: 'long' or 'short'
            
        Returns:
            Dict with 'should_close' and 'reason'
        """
        if position_type == 'long':
            price_change = (current_price - entry_price) / entry_price
        else:
            price_change = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if price_change <= -self.stop_loss_pct:
            return {
                'should_close': True,
                'reason': 'stop_loss',
                'price_change': price_change
            }
        
        # Check take profit
        if price_change >= self.take_profit_pct:
            return {
                'should_close': True,
                'reason': 'take_profit',
                'price_change': price_change
            }
        
        return {
            'should_close': False,
            'reason': 'holding',
            'price_change': price_change
        }
    
    def calculate_kelly_criterion(self,
                                  win_rate: float,
                                  avg_win: float,
                                  avg_loss: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion
        
        Args:
            win_rate: Win rate as decimal (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)
            
        Returns:
            Optimal position size as fraction (0-1)
        """
        if avg_loss == 0 or win_rate == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Use fraction of Kelly for safety (typically 0.25-0.5)
        kelly = max(0, min(kelly * 0.25, self.max_position_size))
        
        return kelly
    
    def get_risk_metrics(self, portfolio_history: np.ndarray) -> Dict[str, float]:
        """
        Calculate various risk metrics
        
        Args:
            portfolio_history: Array of portfolio values over time
            
        Returns:
            Dictionary of risk metrics
        """
        if len(portfolio_history) < 2:
            return {}
        
        # Returns
        returns = np.diff(portfolio_history) / portfolio_history[:-1]
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns, 5) * 100
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(portfolio_history)
        drawdowns = (portfolio_history - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        
        # Current drawdown
        current_dd = drawdowns[-1] * 100
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_dev = np.std(downside_returns) * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_dd,
            'downside_deviation': downside_dev
        }


if __name__ == "__main__":
    # Example usage
    risk_manager = RiskManager()
    
    # Check position size
    is_valid = risk_manager.check_position_size(3000, 10000)
    print(f"Position size valid: {is_valid}")
    
    # Calculate Kelly criterion
    kelly = risk_manager.calculate_kelly_criterion(0.60, 150, 100)
    print(f"Kelly criterion position size: {kelly:.2%}")
