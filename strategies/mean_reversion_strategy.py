import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy that combines volatility and momentum indicators
    to identify potential reversal points in the market.
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize Mean Reversion Strategy
        
        Args:
            params: Dictionary containing strategy parameters:
                bb_period: Bollinger Bands period
                bb_std: Number of standard deviations for Bollinger Bands
                rsi_period: RSI period
                rsi_oversold: RSI oversold threshold
                rsi_overbought: RSI overbought threshold
                volatility_period: Period for volatility calculation
                volume_ma_period: Volume moving average period
                min_volume_ratio: Minimum volume ratio for trade entry
        """
        if params is None:
            params = {}
            
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volatility_period': 20,
            'volume_ma_period': 20,
            'min_volume_ratio': 1.5
        }
        
        # Update default params with provided params
        default_params.update(params)
        super().__init__(default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific technical indicators."""
        df = data.copy()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.params['bb_period']).mean()
        bb_std = df['close'].rolling(window=self.params['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.params['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.params['bb_std'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(
            window=self.params['volatility_period']
        ).std()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=self.params['volume_ma_period']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price distance from BB middle
        df['bb_distance'] = (df['close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_middle'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on mean reversion indicators."""
        df = self.calculate_indicators(data)
        
        # Initialize signals
        df['entry'] = 0
        df['exit'] = False
        
        # Long entry conditions
        long_conditions = (
            (df['close'] <= df['bb_lower']) &  # Price below lower BB
            (df['rsi'] <= self.params['rsi_oversold']) &  # RSI oversold
            (df['volume_ratio'] >= self.params['min_volume_ratio'])  # High volume
        )
        
        # Short entry conditions
        short_conditions = (
            (df['close'] >= df['bb_upper']) &  # Price above upper BB
            (df['rsi'] >= self.params['rsi_overbought']) &  # RSI overbought
            (df['volume_ratio'] >= self.params['min_volume_ratio'])  # High volume
        )
        
        # Set entry signals
        df.loc[long_conditions, 'entry'] = 1
        df.loc[short_conditions, 'entry'] = -1
        
        # Exit conditions
        df.loc[
            (df['entry'].shift(1) == 1) &  # In long position
            ((df['close'] >= df['bb_middle']) |  # Price reaches middle BB
             (df['rsi'] >= 50)),  # RSI crosses above 50
            'exit'
        ] = True
        
        df.loc[
            (df['entry'].shift(1) == -1) &  # In short position
            ((df['close'] <= df['bb_middle']) |  # Price reaches middle BB
             (df['rsi'] <= 50)),  # RSI crosses below 50
            'exit'
        ] = True
        
        return df
