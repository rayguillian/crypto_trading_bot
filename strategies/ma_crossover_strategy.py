import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    Generates buy signals when the fast MA crosses above the slow MA,
    and sell signals when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the MA Crossover strategy.
        
        Default parameters:
            fast_ma_period: 10
            slow_ma_period: 30
            risk_per_trade: 0.02
        """
        default_params = {
            'fast_ma_period': 10,
            'slow_ma_period': 30,
            'risk_per_trade': 0.02
        }
        
        # Update default parameters with provided ones
        if params:
            default_params.update(params)
            
        super().__init__(default_params)
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            bool: True if parameters are valid
        """
        if not all(key in self.params for key in ['fast_ma_period', 'slow_ma_period', 'risk_per_trade']):
            return False
            
        if self.params['fast_ma_period'] >= self.params['slow_ma_period']:
            return False
            
        if not (0 < self.params['risk_per_trade'] <= 0.1):  # Max 10% risk per trade
            return False
            
        return True
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate necessary indicators for the strategy.

        Args:
            data (pd.DataFrame): Market data with OHLCV

        Returns:
            pd.DataFrame: Data with calculated indicators
        """
        data['fast_ma'] = data['close'].rolling(window=self.params['fast_ma_period']).mean()
        data['slow_ma'] = data['close'].rolling(window=self.params['slow_ma_period']).mean()
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV
            
        Returns:
            pd.Series: Trading signals (1: buy, -1: sell, 0: hold)
        """
        data = self.calculate_indicators(data)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Generate crossover signals
        signals[data['fast_ma'] > data['slow_ma']] = 1
        signals[data['fast_ma'] < data['slow_ma']] = -1
        
        # Only keep signal changes
        signals = signals.diff()
        signals = signals.replace(0, np.nan).fillna(0)
        
        return signals
    
    def calculate_position_size(self, data: pd.DataFrame, capital: float) -> float:
        """
        Calculate the position size based on available capital and risk parameters.
        
        Args:
            data (pd.DataFrame): Market data
            capital (float): Available capital
            
        Returns:
            float: Position size
        """
        risk_amount = capital * self.params['risk_per_trade']
        latest_price = data['close'].iloc[-1]
        
        # Calculate stop loss distance (example: 2% from entry)
        stop_loss_distance = latest_price * 0.02
        
        # Calculate position size based on risk
        position_size = risk_amount / stop_loss_distance
        
        return position_size
    
    def get_stop_loss(self, data: pd.DataFrame) -> Optional[float]:
        """
        Calculate stop loss level for current position.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            Optional[float]: Stop loss price level
        """
        if self.position == 0:
            return None
            
        latest_price = data['close'].iloc[-1]
        
        # Example: 2% stop loss
        if self.position > 0:
            return latest_price * 0.98  # Long position stop loss
        else:
            return latest_price * 1.02  # Short position stop loss
    
    def get_take_profit(self, data: pd.DataFrame) -> Optional[float]:
        """
        Calculate take profit level for current position.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            Optional[float]: Take profit price level
        """
        if self.position == 0:
            return None
            
        latest_price = data['close'].iloc[-1]
        
        # Example: 3% take profit (1.5:1 reward-to-risk ratio)
        if self.position > 0:
            return latest_price * 1.03  # Long position take profit
        else:
            return latest_price * 0.97  # Short position take profit
