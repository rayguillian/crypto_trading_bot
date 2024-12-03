import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

class TrendVolumeStrategy(BaseStrategy):
    """
    Enhanced Trend-Volume Strategy with multiple timeframe analysis and advanced filters.
    
    Features:
    - Multi-timeframe trend analysis
    - Volume profile and liquidity analysis
    - Market regime detection
    - Dynamic volatility-adjusted parameters
    - Advanced entry/exit filters
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Enhanced Trend Volume Strategy
        
        Args:
            ema_short: Short EMA period
            ema_medium: Medium EMA period
            ema_long: Long EMA period
            volume_ma_period: Volume moving average period
            atr_period: ATR period for volatility
            min_trend_strength: Minimum trend strength required
            min_volume_percentile: Minimum volume percentile required
            profit_take_atr: Take profit in ATR units
            stop_loss_atr: Stop loss in ATR units
        """
        params = {
            'ema_short': kwargs.get('ema_short', 8),
            'ema_medium': kwargs.get('ema_medium', 21),
            'ema_long': kwargs.get('ema_long', 55),
            'volume_ma_period': kwargs.get('volume_ma_period', 20),
            'atr_period': kwargs.get('atr_period', 14),
            'min_trend_strength': kwargs.get('min_trend_strength', 0.6),
            'min_volume_percentile': kwargs.get('min_volume_percentile', 60),
            'profit_take_atr': kwargs.get('profit_take_atr', 3.0),
            'stop_loss_atr': kwargs.get('stop_loss_atr', 2.0),
            'rsi_period': kwargs.get('rsi_period', 14),
            'rsi_overbought': kwargs.get('rsi_overbought', 70),
            'rsi_oversold': kwargs.get('rsi_oversold', 30),
            'volatility_lookback': kwargs.get('volatility_lookback', 100),
            'trend_lookback': kwargs.get('trend_lookback', 100)
        }
        super().__init__(params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with error handling."""
        try:
            df = data.copy()
            
            # Calculate EMAs
            df['ema_short'] = df['close'].ewm(span=self.params['ema_short'], adjust=False).mean()
            df['ema_medium'] = df['close'].ewm(span=self.params['ema_medium'], adjust=False).mean()
            df['ema_long'] = df['close'].ewm(span=self.params['ema_long'], adjust=False).mean()
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=self.params['atr_period']).mean()
            
            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(window=self.params['volume_ma_period']).mean()
            df['volume_std'] = df['volume'].rolling(window=self.params['volume_ma_period']).std()
            df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume_std']
            df['volume_percentile'] = df['volume'].rolling(window=self.params['volume_ma_period']).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
            
            # Trend strength indicators
            df['returns'] = df['close'].pct_change()
            df['trend_returns'] = df['returns'].rolling(window=self.params['trend_lookback']).mean()
            df['trend_std'] = df['returns'].rolling(window=self.params['trend_lookback']).std()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volatility analysis
            df['volatility'] = df['returns'].rolling(window=self.params['volatility_lookback']).std() * np.sqrt(252)
            df['volatility_ma'] = df['volatility'].rolling(window=self.params['volume_ma_period']).mean()
            df['volatility_ratio'] = df['volatility'] / df['volatility_ma']
            
            # Trend strength calculation
            df['trend_strength'] = self._calculate_trend_strength(df)
            
            # Market regime
            df['regime'] = self._detect_market_regime(df)
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return data
            
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with comprehensive filtering."""
        try:
            df = self.calculate_indicators(data)
            
            # Initialize signal columns
            df['entry'] = 0
            df['exit'] = False
            
            # Calculate trend conditions
            uptrend = (
                (df['ema_short'] > df['ema_medium']) &
                (df['ema_medium'] > df['ema_long']) &
                (df['close'] > df['ema_short']) &
                (df['trend_strength'] > self.params['min_trend_strength'])
            )
            
            downtrend = (
                (df['ema_short'] < df['ema_medium']) &
                (df['ema_medium'] < df['ema_long']) &
                (df['close'] < df['ema_short']) &
                (df['trend_strength'] > self.params['min_trend_strength'])
            )
            
            # Volume conditions
            volume_filter = (
                (df['volume_percentile'] > self.params['min_volume_percentile'] / 100) &
                (df['volume_zscore'] > 0)
            )
            
            # Volatility filters
            normal_volatility = (df['volatility_ratio'] < 1.5)  # Avoid excessive volatility
            
            # RSI filters
            rsi_filter_long = (df['rsi'] < self.params['rsi_overbought'])
            rsi_filter_short = (df['rsi'] > self.params['rsi_oversold'])
            
            # Combine entry conditions
            long_entry = (
                uptrend &
                volume_filter &
                normal_volatility &
                rsi_filter_long &
                (df['regime'] != 'volatile')  # Avoid volatile regimes
            )
            
            short_entry = (
                downtrend &
                volume_filter &
                normal_volatility &
                rsi_filter_short &
                (df['regime'] != 'volatile')
            )
            
            # Set entry signals
            df.loc[long_entry, 'entry'] = 1
            df.loc[short_entry, 'entry'] = -1
            
            # Calculate exit conditions
            df['exit'] = self._calculate_exit_signals(df)
            
            return df
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return data
            
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using multiple metrics."""
        try:
            # Linear regression slope
            def calculate_slope(prices):
                if len(prices) < 2:
                    return 0
                x = np.arange(len(prices))
                slope, _, r_value, _, _ = linregress(x, prices)
                return abs(slope * r_value**2)
                
            # Calculate rolling slope
            slopes = df['close'].rolling(window=self.params['trend_lookback']).apply(
                calculate_slope, raw=False
            )
            
            # Normalize slopes
            scaler = StandardScaler()
            normalized_slopes = scaler.fit_transform(slopes.values.reshape(-1, 1)).flatten()
            
            # Convert to 0-1 range
            trend_strength = pd.Series(normalized_slopes).clip(lower=0)
            trend_strength = trend_strength / trend_strength.max() if trend_strength.max() > 0 else trend_strength
            
            return trend_strength
            
        except Exception as e:
            print(f"Error calculating trend strength: {e}")
            return pd.Series(0, index=df.index)
            
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime using volatility and trend metrics."""
        try:
            regimes = pd.Series(index=df.index, data='normal')
            
            # Volatility based regime
            high_vol = df['volatility'] > df['volatility'].quantile(0.8)
            low_vol = df['volatility'] < df['volatility'].quantile(0.2)
            
            # Trend based regime
            strong_trend = df['trend_strength'] > 0.7
            weak_trend = df['trend_strength'] < 0.3
            
            # Classify regimes
            regimes[high_vol] = 'volatile'
            regimes[low_vol & weak_trend] = 'ranging'
            regimes[strong_trend & ~high_vol] = 'trending'
            
            return regimes
            
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return pd.Series('normal', index=df.index)
            
    def _calculate_exit_signals(self, df: pd.DataFrame) -> pd.Series:
        """Calculate exit signals based on multiple conditions."""
        try:
            exits = pd.Series(False, index=df.index)
            
            # Trend reversal exits
            trend_reversal = (
                (df['ema_short'].shift(1) > df['ema_medium'].shift(1)) &
                (df['ema_short'] < df['ema_medium'])
            ) | (
                (df['ema_short'].shift(1) < df['ema_medium'].shift(1)) &
                (df['ema_short'] > df['ema_medium'])
            )
            
            # Volatility based exits
            volatility_exit = df['volatility_ratio'] > 2.0
            
            # RSI extreme exits
            rsi_exit = (df['rsi'] > self.params['rsi_overbought']) | (df['rsi'] < self.params['rsi_oversold'])
            
            # Combine exit conditions
            exits = trend_reversal | volatility_exit | rsi_exit
            
            return exits
            
        except Exception as e:
            print(f"Error calculating exit signals: {e}")
            return pd.Series(False, index=df.index)
            
    def calculate_position_size(self, 
                              equity: float,
                              entry_price: float,
                              atr: float,
                              volatility_ratio: float) -> Tuple[float, float, float]:
        """
        Calculate position size and risk parameters.
        
        Returns:
            Tuple of (position_size, stop_loss_price, take_profit_price)
        """
        try:
            # Adjust ATR multipliers based on volatility
            stop_loss_multiplier = self.params['stop_loss_atr'] * (1 + (volatility_ratio - 1) * 0.5)
            take_profit_multiplier = self.params['profit_take_atr'] * (1 + (volatility_ratio - 1) * 0.5)
            
            # Calculate stop and target distances
            stop_distance = atr * stop_loss_multiplier
            target_distance = atr * take_profit_multiplier
            
            # Risk 1% of equity per trade, adjusted for volatility
            risk_amount = equity * 0.01 * (1 / volatility_ratio)
            
            # Calculate position size
            position_size = risk_amount / stop_distance
            
            # Calculate prices
            stop_loss_price = entry_price - stop_distance
            take_profit_price = entry_price + target_distance
            
            return position_size, stop_loss_price, take_profit_price
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.0, entry_price * 0.99, entry_price * 1.01
