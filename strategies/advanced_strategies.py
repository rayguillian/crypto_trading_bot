import numpy as np
import pandas as pd
from typing import Dict, Any
from strategies.base_strategy import BaseStrategy
import talib

class BollingerRSIStrategy(BaseStrategy):
    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_high: float = 70,
                 rsi_low: float = 30,
                 volume_factor: float = 1.5):
        super().__init__()
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_high = rsi_high
        self.rsi_low = rsi_low
        self.volume_factor = volume_factor

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate Bollinger Bands
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(
            data['close'], timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        
        # Calculate RSI
        data['RSI'] = talib.RSI(data['close'], timeperiod=self.rsi_period)
        
        # Calculate Volume MA
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = 0
        signals['exit'] = False
        
        # Long signals
        signals.loc[
            (data['close'] <= data['BB_lower']) &  # Price below lower band
            (data['RSI'] < self.rsi_low) &  # Oversold
            (data['volume'] > data['volume_ma'] * self.volume_factor),  # High volume
            'entry'
        ] = 1
        
        # Short signals
        signals.loc[
            (data['close'] >= data['BB_upper']) &  # Price above upper band
            (data['RSI'] > self.rsi_high) &  # Overbought
            (data['volume'] > data['volume_ma'] * self.volume_factor),  # High volume
            'entry'
        ] = -1
        
        # Exit signals
        signals['exit'] = (
            ((data['close'] > data['BB_middle']) & (signals['entry'].shift(1) == 1)) |  # Exit long
            ((data['close'] < data['BB_middle']) & (signals['entry'].shift(1) == -1))   # Exit short
        )
        
        return signals

class TrendVolumeStrategy(BaseStrategy):
    def __init__(self,
                 ema_short: int = 10,
                 ema_medium: int = 20,
                 ema_long: int = 50,
                 volume_ma: int = 20,
                 volume_mult: float = 1.5,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        super().__init__()
        self.ema_short = ema_short
        self.ema_medium = ema_medium
        self.ema_long = ema_long
        self.volume_ma = volume_ma
        self.volume_mult = volume_mult
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate EMAs
        data['EMA_short'] = talib.EMA(data['close'], timeperiod=self.ema_short)
        data['EMA_medium'] = talib.EMA(data['close'], timeperiod=self.ema_medium)
        data['EMA_long'] = talib.EMA(data['close'], timeperiod=self.ema_long)
        
        # Calculate Volume MA
        data['volume_ma'] = talib.SMA(data['volume'], timeperiod=self.volume_ma)
        
        # Calculate ATR
        data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=self.atr_period)
        
        # Calculate trend strength
        data['trend_strength'] = (
            (data['EMA_short'] - data['EMA_long']).abs() / data['ATR']
        )
        
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = 0
        signals['exit'] = False
        
        # Long signals
        signals.loc[
            (data['EMA_short'] > data['EMA_medium']) &
            (data['EMA_medium'] > data['EMA_long']) &
            (data['volume'] > data['volume_ma'] * self.volume_mult) &
            (data['trend_strength'] > self.atr_multiplier),
            'entry'
        ] = 1
        
        # Short signals
        signals.loc[
            (data['EMA_short'] < data['EMA_medium']) &
            (data['EMA_medium'] < data['EMA_long']) &
            (data['volume'] > data['volume_ma'] * self.volume_mult) &
            (data['trend_strength'] > self.atr_multiplier),
            'entry'
        ] = -1
        
        # Exit signals
        signals['exit'] = (
            ((data['EMA_short'] < data['EMA_medium']) & (signals['entry'].shift(1) == 1)) |
            ((data['EMA_short'] > data['EMA_medium']) & (signals['entry'].shift(1) == -1))
        )
        
        return signals

class MachineLearningStrategy(BaseStrategy):
    def __init__(self,
                 lookback_period: int = 20,
                 prediction_threshold: float = 0.6,
                 feature_window: int = 10):
        super().__init__()
        self.lookback_period = lookback_period
        self.prediction_threshold = prediction_threshold
        self.feature_window = feature_window
        self.model = None

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Technical indicators
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'return_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'return_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
        
        return df.dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Create features
        features_df = self._create_features(data)
        
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = 0
        signals['exit'] = False
        
        # Generate signals based on feature combinations
        signals.loc[
            (features_df['RSI'] < 30) &
            (features_df['MACD'] > features_df['MACD_signal']) &
            (features_df['volume_ratio'] > 1.5) &
            (features_df['return_mean_5'] > 0),
            'entry'
        ] = 1
        
        signals.loc[
            (features_df['RSI'] > 70) &
            (features_df['MACD'] < features_df['MACD_signal']) &
            (features_df['volume_ratio'] > 1.5) &
            (features_df['return_mean_5'] < 0),
            'entry'
        ] = -1
        
        # Exit signals
        signals['exit'] = (
            ((features_df['RSI'] > 50) & (signals['entry'].shift(1) == 1)) |
            ((features_df['RSI'] < 50) & (signals['entry'].shift(1) == -1))
        )
        
        return signals

class AdvancedStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.ma_fast = kwargs.get('ma_fast', 20)
        self.ma_slow = kwargs.get('ma_slow', 50)
        self.volume_threshold = kwargs.get('volume_threshold', 1.5)
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate technical indicators
        data['RSI'] = talib.RSI(data['close'], timeperiod=self.rsi_period)
        data['MA_fast'] = talib.SMA(data['close'], timeperiod=self.ma_fast)
        data['MA_slow'] = talib.SMA(data['close'], timeperiod=self.ma_slow)
        data['Volume_MA'] = talib.SMA(data['volume'], timeperiod=20)
        data['Volume_Ratio'] = data['volume'] / data['Volume_MA']
        
        # Initialize signals
        data['signal'] = 0
        
        # Generate buy signals
        buy_condition = (
            (data['RSI'] < 30) &  # Oversold
            (data['MA_fast'] > data['MA_slow']) &  # Uptrend
            (data['Volume_Ratio'] > self.volume_threshold)  # High volume
        )
        data.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals
        sell_condition = (
            (data['RSI'] > 70) |  # Overbought
            (data['MA_fast'] < data['MA_slow'])  # Downtrend
        )
        data.loc[sell_condition, 'signal'] = -1
        
        return data
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.rsi_period = params.get('rsi_period', self.rsi_period)
        self.ma_fast = params.get('ma_fast', self.ma_fast)
        self.ma_slow = params.get('ma_slow', self.ma_slow)
        self.volume_threshold = params.get('volume_threshold', self.volume_threshold)
