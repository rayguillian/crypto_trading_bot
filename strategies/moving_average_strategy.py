import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        """
        Simple moving average crossover strategy with RSI filter.
        
        Args:
            short_window: Short-term moving average period
            long_window: Long-term moving average period
            rsi_period: RSI calculation period
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
        """
        params = {
            'short_window': kwargs.get('short_window', 20),
            'long_window': kwargs.get('long_window', 50),
            'rsi_period': kwargs.get('rsi_period', 14),
            'rsi_overbought': kwargs.get('rsi_overbought', 70),
            'rsi_oversold': kwargs.get('rsi_oversold', 30)
        }
        super().__init__(params)
        
    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy-specific technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        df = data.copy()
        
        # Calculate moving averages
        df['short_ma'] = df['close'].rolling(window=self.params['short_window']).mean()
        df['long_ma'] = df['close'].rolling(window=self.params['long_window']).mean()
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover and RSI.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with entry and exit signals
        """
        df = self.calculate_indicators(data)
        
        # Initialize signals
        df['entry'] = 0
        df['exit'] = False
        
        # Generate signals
        # Buy when short MA crosses above long MA and RSI is oversold
        buy_condition = (
            (df['short_ma'] > df['long_ma']) & 
            (df['short_ma'].shift(1) <= df['long_ma'].shift(1)) &
            (df['rsi'] < self.params['rsi_oversold'])
        )
        
        # Sell when short MA crosses below long MA or RSI is overbought
        sell_condition = (
            (df['short_ma'] < df['long_ma']) & 
            (df['short_ma'].shift(1) >= df['long_ma'].shift(1)) |
            (df['rsi'] > self.params['rsi_overbought'])
        )
        
        # Set entry signals (1 for long, -1 for short)
        df.loc[buy_condition, 'entry'] = 1
        df.loc[sell_condition, 'entry'] = -1
        
        # Set exit signals
        # Exit long positions when sell condition occurs
        df.loc[sell_condition & (df['entry'].shift(1) >= 0), 'exit'] = True
        # Exit short positions when buy condition occurs
        df.loc[buy_condition & (df['entry'].shift(1) <= 0), 'exit'] = True
        
        return df
    
    def get_parameters(self) -> Dict:
        """Return current strategy parameters."""
        return self.params
    
    def set_parameters(self, parameters: Dict):
        """Set strategy parameters."""
        self.params.update(parameters)
