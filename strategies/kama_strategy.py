import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
from ta.momentum import KAMAIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import logging

class KAMAStrategy(BaseStrategy):
    """
    Kaufman's Adaptive Moving Average (KAMA) Strategy with multiple technical indicators
    for trend confirmation and risk management.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the KAMA strategy with default or custom parameters.
        """
        default_params = {
            'kama_window': 14,
            'kama_fast': 2,
            'kama_slow': 30,
            'atr_period': 14,
            'adx_period': 14,
            'adx_threshold': 25.0,
            'chop_period': 14,
            'chop_threshold': 50.0,
            'bbw_threshold': 7.0,
            'cooldown_periods': 10,
            'atr_multiplier_stop': 2.5,
            'atr_multiplier_tp': 2.5,
            'risk_percent': 1.0,
            'min_position_size': 0.001
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(default_params)
        self.last_signal_index = -self.parameters['cooldown_periods']
        
    def prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for the strategy.
        """
        df = data.copy()
        
        # Calculate KAMA
        kama = KAMAIndicator(
            close=df['close'],
            window=self.parameters['kama_window'],
            pow1=self.parameters['kama_fast'],
            pow2=self.parameters['kama_slow']
        )
        df['kama'] = kama.kama()
        
        # Calculate ADX
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.parameters['adx_period']
        )
        df['adx'] = adx.adx()
        
        # Calculate ATR
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.parameters['atr_period']
        )
        df['atr'] = atr.average_true_range()
        
        # Calculate Choppiness Index
        high_low_range = df['high'] - df['low']
        sum_range = high_low_range.rolling(window=self.parameters['chop_period']).sum()
        max_high = df['high'].rolling(window=self.parameters['chop_period']).max()
        min_low = df['low'].rolling(window=self.parameters['chop_period']).min()
        df['chop'] = 100 * np.log10(sum_range / (max_high - min_low)) / np.log10(self.parameters['chop_period'])
        
        # Calculate Bollinger Band Width
        bb = BollingerBands(
            close=df['close'],
            window=20,
            window_dev=2
        )
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on KAMA and other indicators.
        """
        df = self.prepare_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            # Skip if in cooldown period
            if i - self.last_signal_index < self.parameters['cooldown_periods']:
                continue
                
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Check trend conditions
            trend_up = current['close'] > current['kama']
            trend_down = current['close'] < current['kama']
            
            # Check additional conditions
            strong_trend = current['adx'] > self.parameters['adx_threshold']
            low_chop = current['chop'] < self.parameters['chop_threshold']
            low_volatility = current['bb_width'] < self.parameters['bbw_threshold']
            
            # Generate signals
            if trend_up and strong_trend and low_chop and low_volatility:
                signals.iloc[i] = 1
                self.last_signal_index = i
            elif trend_down and strong_trend and low_chop and low_volatility:
                signals.iloc[i] = -1
                self.last_signal_index = i
                
        return signals
        
    def calculate_position_size(self, data: pd.DataFrame, capital: float) -> float:
        """
        Calculate position size based on ATR and risk parameters.
        """
        df = self.prepare_indicators(data)
        current_atr = df['atr'].iloc[-1]
        
        if current_atr == 0:
            return self.parameters['min_position_size']
            
        risk_amount = capital * (self.parameters['risk_percent'] / 100)
        position_size = risk_amount / (current_atr * self.parameters['atr_multiplier_stop'])
        
        return max(position_size, self.parameters['min_position_size'])
        
    def get_stop_loss(self, data: pd.DataFrame) -> Optional[float]:
        """
        Calculate stop loss level based on ATR.
        """
        if self.position == 0:
            return None
            
        df = self.prepare_indicators(data)
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        if self.position > 0:  # Long position
            return current_price - (current_atr * self.parameters['atr_multiplier_stop'])
        else:  # Short position
            return current_price + (current_atr * self.parameters['atr_multiplier_stop'])
            
    def get_take_profit(self, data: pd.DataFrame) -> Optional[float]:
        """
        Calculate take profit level based on ATR.
        """
        if self.position == 0:
            return None
            
        df = self.prepare_indicators(data)
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        if self.position > 0:  # Long position
            return current_price + (current_atr * self.parameters['atr_multiplier_tp'])
        else:  # Short position
            return current_price - (current_atr * self.parameters['atr_multiplier_tp'])
            
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        """
        required_params = [
            'kama_window', 'kama_fast', 'kama_slow', 'atr_period',
            'adx_period', 'adx_threshold', 'chop_period', 'chop_threshold',
            'bbw_threshold', 'cooldown_periods', 'atr_multiplier_stop',
            'atr_multiplier_tp', 'risk_percent', 'min_position_size'
        ]
        
        # Check if all required parameters are present
        if not all(param in self.parameters for param in required_params):
            return False
            
        # Validate parameter values
        if (self.parameters['kama_fast'] >= self.parameters['kama_slow'] or
            self.parameters['risk_percent'] <= 0 or
            self.parameters['risk_percent'] > 5 or  # Max 5% risk per trade
            self.parameters['min_position_size'] <= 0):
            return False
            
        return True
