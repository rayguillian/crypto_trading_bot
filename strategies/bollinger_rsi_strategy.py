import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .base_strategy import BaseStrategy

class BollingerRSIStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        """
        Initialize Bollinger Bands + RSI Strategy
        
        Args:
            bb_period: Period for Bollinger Bands calculation
            bb_std: Number of standard deviations for Bollinger Bands
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            volume_factor: Volume increase factor to confirm signals
        """
        params = {
            'bb_period': kwargs.get('bb_period', 20),
            'bb_std': kwargs.get('bb_std', 2.0),
            'rsi_period': kwargs.get('rsi_period', 14),
            'rsi_overbought': kwargs.get('rsi_overbought', 70),
            'rsi_oversold': kwargs.get('rsi_oversold', 30),
            'volume_factor': kwargs.get('volume_factor', 1.5)
        }
        super().__init__(params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and RSI indicators."""
        # Calculate Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=self.params['bb_period']).mean()
        data['bb_std'] = data['close'].rolling(window=self.params['bb_period']).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * self.params['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * self.params['bb_std'])
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volume MA for confirmation
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Bollinger Bands and RSI."""
        data = self.calculate_indicators(data)
        
        # Initialize signals
        data['entry'] = 0
        data['exit'] = False
        
        # Buy conditions
        buy_conditions = (
            (data['close'] < data['bb_lower']) &  # Price below lower band
            (data['rsi'] < self.params['rsi_oversold']) &   # RSI oversold
            (data['volume'] > data['volume_ma'] * self.params['volume_factor'])  # Volume confirmation
        )
        
        # Sell conditions
        sell_conditions = (
            (data['close'] > data['bb_upper']) &  # Price above upper band
            (data['rsi'] > self.params['rsi_overbought']) & # RSI overbought
            (data['volume'] > data['volume_ma'] * self.params['volume_factor'])  # Volume confirmation
        )
        
        # Set entry signals
        data.loc[buy_conditions, 'entry'] = 1
        data.loc[sell_conditions, 'entry'] = -1
        
        # Set exit signals
        # Exit long positions when sell condition occurs
        data.loc[sell_conditions & (data['entry'].shift(1) >= 0), 'exit'] = True
        # Exit short positions when buy condition occurs
        data.loc[buy_conditions & (data['entry'].shift(1) <= 0), 'exit'] = True
        
        return data
    
    def get_parameters(self) -> Dict:
        """Return current strategy parameters."""
        return self.params
    
    def set_parameters(self, parameters: Dict):
        """Set strategy parameters."""
        self.params.update(parameters)
