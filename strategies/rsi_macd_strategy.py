import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import ta
from .base_strategy import BaseStrategy, Signal

class RSIMACDStrategy(BaseStrategy):
    """
    A strategy combining RSI and MACD indicators with volatility adaptation.
    
    Parameters:
    - rsi_period: Period for RSI calculation (default: 21)
    - rsi_overbought: RSI overbought threshold (default: 65)
    - rsi_oversold: RSI oversold threshold (default: 35)
    - macd_fast: Fast period for MACD (default: 10)
    - macd_slow: Slow period for MACD (default: 20)
    - macd_signal: Signal period for MACD (default: 9)
    - atr_period: Period for ATR calculation (default: 14)
    - atr_multiplier: Multiplier for ATR-based stops (default: 2.5)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize strategy with default or custom parameters."""
        default_params = self.get_default_parameters()
        
        if params:
            default_params.update(params)
            
        super().__init__(default_params)
        
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        """Get default strategy parameters."""
        return {
            'rsi_period': 21,
            'rsi_overbought': 65,
            'rsi_oversold': 35,
            'macd_fast': 10,
            'macd_slow': 20,
            'macd_signal': 9,
            'atr_period': 14,
            'atr_multiplier': 2.5
        }
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI, MACD, and ATR indicators."""
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'],
            window=self.params['rsi_period']
        ).rsi()
        
        # Calculate MACD
        macd = ta.trend.MACD(
            close=df['close'],
            window_fast=self.params['macd_fast'],
            window_slow=self.params['macd_slow'],
            window_sign=self.params['macd_signal']
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Calculate ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.params['atr_period']
        ).average_true_range()
        
        # Calculate volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df
        
    def get_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Get trading signal for current market conditions.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            current_idx: Current index in the DataFrame
            
        Returns:
            Signal object with entry/exit decisions
        """
        df = self.calculate_indicators(data.iloc[:current_idx + 1])
        current = df.iloc[-1]
        
        # Initialize signal
        entry = 0.0
        exit_signal = False
        
        # Generate entry signals
        if current['rsi'] < self.params['rsi_oversold'] and current['macd_diff'] > 0 and current['macd'] < 0:
            entry = 1.0
        elif current['rsi'] > self.params['rsi_overbought'] and current['macd_diff'] < 0 and current['macd'] > 0:
            entry = -1.0
            
        # Generate exit signals
        if (entry > 0 and (current['rsi'] > self.params['rsi_overbought'] or current['macd_diff'] < 0)) or \
           (entry < 0 and (current['rsi'] < self.params['rsi_oversold'] or current['macd_diff'] > 0)):
            exit_signal = True
            
        return Signal(
            entry=entry,
            exit=exit_signal,
            stop_loss=self._calculate_stop_loss(data, current_idx),
            take_profit=self._calculate_take_profit(data, current_idx),
            position_size=self._calculate_position_size(data, current_idx)
        )
        
    def _calculate_stop_loss(self, data: pd.DataFrame, current_idx: int) -> Optional[float]:
        """Calculate stop loss using ATR."""
        df = self.calculate_indicators(data.iloc[:current_idx + 1])
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        return current_price - (self.params['atr_multiplier'] * atr)
        
    def _calculate_take_profit(self, data: pd.DataFrame, current_idx: int) -> Optional[float]:
        """Calculate take profit using ATR."""
        df = self.calculate_indicators(data.iloc[:current_idx + 1])
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Use 1.5x the stop loss distance for take profit
        return current_price + (self.params['atr_multiplier'] * 1.5 * atr)
        
    def _calculate_position_size(self, data: pd.DataFrame, current_idx: int) -> Optional[float]:
        """Calculate position size based on volatility."""
        df = self.calculate_indicators(data.iloc[:current_idx + 1])
        volatility = df['volatility'].iloc[-1]
        
        # Reduce position size as volatility increases
        base_size = 1.0
        vol_factor = 1 - min(volatility * 10, 0.5)  # Cap at 50% reduction
        
        return base_size * vol_factor
        
    def get_required_indicators(self) -> list:
        """Get list of required indicators."""
        return ['rsi', 'macd', 'macd_signal', 'macd_diff', 'atr', 'volatility']
        
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        valid = True
        
        # Check RSI parameters
        valid &= 5 <= self.params['rsi_period'] <= 50
        valid &= 50 <= self.params['rsi_overbought'] <= 90
        valid &= 10 <= self.params['rsi_oversold'] <= 50
        valid &= self.params['rsi_oversold'] < self.params['rsi_overbought']
        
        # Check MACD parameters
        valid &= 5 <= self.params['macd_fast'] <= 50
        valid &= 10 <= self.params['macd_slow'] <= 100
        valid &= 5 <= self.params['macd_signal'] <= 50
        valid &= self.params['macd_fast'] < self.params['macd_slow']
        
        # Check ATR parameters
        valid &= 5 <= self.params['atr_period'] <= 50
        valid &= 0.5 <= self.params['atr_multiplier'] <= 5.0
        
        return valid

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        
        # Initialize signal columns
        signals['entry'] = 0.0
        signals['exit'] = False
        
        # Generate entry signals
        long_condition = (
            (df['rsi'] < self.params['rsi_oversold']) & 
            (df['macd_diff'] > 0) & 
            (df['macd'] < 0)
        )
        short_condition = (
            (df['rsi'] > self.params['rsi_overbought']) & 
            (df['macd_diff'] < 0) & 
            (df['macd'] > 0)
        )
        
        signals.loc[long_condition, 'entry'] = 1.0
        signals.loc[short_condition, 'entry'] = -1.0
        
        # Generate exit signals
        long_exit = (
            (df['rsi'] > self.params['rsi_overbought']) | 
            (df['macd_diff'] < 0)
        )
        short_exit = (
            (df['rsi'] < self.params['rsi_oversold']) | 
            (df['macd_diff'] > 0)
        )
        
        signals.loc[long_exit | short_exit, 'exit'] = True
        
        # Add stop loss and take profit levels
        signals['stop_loss'] = df['close'] - (df['atr'] * self.params['atr_multiplier'])
        signals['take_profit'] = df['close'] + (df['atr'] * self.params['atr_multiplier'] * 1.5)
        
        return signals
