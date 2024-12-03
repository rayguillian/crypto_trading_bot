import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import logging

class BBandsVolumeStrategy(BaseStrategy):
    """
    A strategy combining Bollinger Bands with Volume Profile analysis.
    
    Entry Conditions:
    - Long: Price touches lower band with increasing volume AND price > VWAP
    - Short: Price touches upper band with increasing volume AND price < VWAP
    
    Exit Conditions:
    - Long: Price crosses middle band downward OR volume decreases significantly
    - Short: Price crosses middle band upward OR volume decreases significantly
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize strategy with parameters."""
        default_params = {
            'bb_window': 20,
            'bb_std': 2.0,
            'volume_window': 20,
            'volume_threshold': 1.5,
            'vwap_window': 14,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'risk_per_trade': 1.0,
            'min_position_size': 0.001
        }
        
        super().__init__(parameters or default_params)
    
    def prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        df = data.copy()
        
        # Calculate Bollinger Bands
        bb = BollingerBands(
            close=df['close'],
            window=self.parameters['bb_window'],
            window_dev=self.parameters['bb_std']
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Calculate VWAP
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=self.parameters['vwap_window']
        )
        df['vwap'] = vwap.volume_weighted_average_price()
        
        # Calculate volume metrics
        df['volume_sma'] = df['volume'].rolling(window=self.parameters['volume_window']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Calculate ATR for position sizing
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.parameters['atr_period']).mean()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        df = self.prepare_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            # Skip if not enough data for indicators
            if pd.isna(df['bb_upper'].iloc[i]) or pd.isna(df['vwap'].iloc[i]):
                continue
            
            price = df['close'].iloc[i]
            volume_ratio = df['volume_ratio'].iloc[i]
            vwap = df['vwap'].iloc[i]
            
            # Long entry conditions
            if (price <= df['bb_lower'].iloc[i] and 
                volume_ratio > self.parameters['volume_threshold'] and
                price > vwap):
                signals.iloc[i] = 1
            
            # Short entry conditions
            elif (price >= df['bb_upper'].iloc[i] and 
                  volume_ratio > self.parameters['volume_threshold'] and
                  price < vwap):
                signals.iloc[i] = -1
        
        return signals
    
    def calculate_position_size(self, data: pd.DataFrame, capital: float) -> float:
        """Calculate position size based on ATR and risk parameters."""
        if len(data) < self.parameters['atr_period']:
            return self.parameters['min_position_size']
        
        current_price = data['close'].iloc[-1]
        current_atr = data['atr'].iloc[-1]
        
        # Calculate position size based on risk
        risk_amount = capital * (self.parameters['risk_per_trade'] / 100)
        stop_distance = current_atr * self.parameters['atr_multiplier']
        
        if stop_distance == 0:
            return self.parameters['min_position_size']
        
        position_size = risk_amount / stop_distance
        
        # Ensure minimum position size
        return max(position_size, self.parameters['min_position_size'])
    
    def get_stop_loss(self, data: pd.DataFrame) -> float:
        """Calculate stop loss level based on ATR and Bollinger Bands."""
        if len(data) < self.parameters['atr_period']:
            return None
        
        current_price = data['close'].iloc[-1]
        current_atr = data['atr'].iloc[-1]
        bb_middle = data['bb_middle'].iloc[-1]
        
        if self.position > 0:  # Long position
            return min(
                current_price - (current_atr * self.parameters['atr_multiplier']),
                bb_middle
            )
        elif self.position < 0:  # Short position
            return max(
                current_price + (current_atr * self.parameters['atr_multiplier']),
                bb_middle
            )
        
        return None
    
    def get_take_profit(self, data: pd.DataFrame) -> float:
        """Calculate take profit level based on Bollinger Bands."""
        if len(data) < self.parameters['bb_window']:
            return None
        
        if self.position > 0:  # Long position
            return data['bb_upper'].iloc[-1]
        elif self.position < 0:  # Short position
            return data['bb_lower'].iloc[-1]
        
        return None
