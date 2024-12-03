from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class Signal:
    entry: float  # 1 for long, -1 for short, 0 for no entry
    exit: bool
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize strategy with parameters.
        
        Args:
            params: Dictionary of strategy parameters
        """
        self.params = params
        self.position = None
        self.signals = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        pass
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy-specific technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        pass
        
    def get_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Get trading signal for current market conditions.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            current_idx: Current index in the DataFrame
            
        Returns:
            Signal object with entry/exit decisions
        """
        signals_df = self.generate_signals(data.iloc[:current_idx + 1])
        current_signal = signals_df.iloc[-1]
        
        return Signal(
            entry=current_signal['entry'],
            exit=current_signal['exit'],
            stop_loss=self._calculate_stop_loss(data, current_idx),
            take_profit=self._calculate_take_profit(data, current_idx),
            position_size=self._calculate_position_size(data, current_idx)
        )
        
    def _calculate_stop_loss(self, data: pd.DataFrame, current_idx: int) -> Optional[float]:
        """Calculate stop loss price based on market conditions."""
        if 'atr' not in data.columns:
            return None
            
        current_price = data['close'].iloc[current_idx]
        atr = data['atr'].iloc[current_idx]
        
        # Default to 2 ATR for stop loss
        return current_price - (2 * atr)
        
    def _calculate_take_profit(self, data: pd.DataFrame, current_idx: int) -> Optional[float]:
        """Calculate take profit price based on market conditions."""
        if 'atr' not in data.columns:
            return None
            
        current_price = data['close'].iloc[current_idx]
        atr = data['atr'].iloc[current_idx]
        
        # Default to 3 ATR for take profit (1:1.5 risk-reward)
        return current_price + (3 * atr)
        
    def _calculate_position_size(self, data: pd.DataFrame, current_idx: int) -> Optional[float]:
        """Calculate suggested position size based on volatility."""
        if 'volatility' not in data.columns:
            return None
            
        volatility = data['volatility'].iloc[current_idx]
        
        # Adjust position size inversely to volatility
        base_size = 1.0
        return base_size * (1 - min(volatility, 0.5))
        
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        return True  # Override in child classes
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]):
        """Update strategy parameters."""
        self.params.update(params)
        
    def get_required_indicators(self) -> list:
        """Get list of required indicators for the strategy."""
        return ['atr', 'volatility']  # Override in child classes
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state."""
        return {
            'name': self.__class__.__name__,
            'parameters': self.params,
            'position': self.position,
            'required_indicators': self.get_required_indicators()
        }
