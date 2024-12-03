import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy.stats import zscore
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from strategies.base_strategy import BaseStrategy

class FeatureEngineer:
    """Advanced feature engineering with machine learning components."""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50, 100]):
        self.lookback_periods = lookback_periods
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.regime_classifier = KMeans(n_clusters=4, random_state=42)
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced technical features."""
        df = data.copy()
        
        # Basic price features
        for period in self.lookback_periods:
            # Returns and volatility
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df[f'returns_{period}'].rolling(period).std()
            
            # Price momentum
            df[f'momentum_{period}'] = df['close'].div(df['close'].shift(period)).sub(1)
            
            # Volume momentum
            df[f'volume_momentum_{period}'] = df['volume'].div(df['volume'].shift(period)).sub(1)
            
            # Price acceleration
            df[f'acceleration_{period}'] = df[f'momentum_{period}'].sub(df[f'momentum_{period}'].shift(1))
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Advanced ML features
        df = self._add_ml_features(df)
        
        # Clean and scale features
        df = self._clean_and_scale_features(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        try:
            # Ensure we have numeric data and handle any conversion errors
            for col in ['high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Forward fill any NaN values from the conversion
            df = df.ffill().bfill()
            
            # Replace any infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()
            
            # Verify data quality before proceeding
            if df[['high', 'low', 'close', 'volume']].isnull().any().any():
                raise ValueError("Critical data contains NaN values after cleaning")
            
            # Calculate indicators in order of dependency
            
            # 1. Trend indicators
            try:
                adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
                df['adx'] = adx.adx()
                df['di_plus'] = adx.adx_pos()
                df['di_minus'] = adx.adx_neg()
                
                # Fill ADX with neutral values
                df['adx'] = df['adx'].fillna(50)
                df['di_plus'] = df['di_plus'].fillna(0)
                df['di_minus'] = df['di_minus'].fillna(0)
            except Exception as e:
                print(f"Warning: Error calculating ADX: {str(e)}")
                df['adx'] = 50
                df['di_plus'] = 0
                df['di_minus'] = 0
            
            # 2. Moving averages
            for period in self.lookback_periods:
                try:
                    ema = EMAIndicator(close=df['close'], window=period)
                    df[f'ema_{period}'] = ema.ema_indicator()
                    df[f'ema_{period}'] = df[f'ema_{period}'].ffill().fillna(df['close'])
                except Exception as e:
                    print(f"Warning: Error calculating EMA for period {period}: {str(e)}")
                    df[f'ema_{period}'] = df['close']
            
            # 3. Momentum indicators
            try:
                rsi = RSIIndicator(close=df['close'])
                df['rsi'] = rsi.rsi()
                df['rsi'] = df['rsi'].fillna(50)
                
                stoch_rsi = StochRSIIndicator(close=df['close'])
                df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
                df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
                
                df['stoch_rsi_k'] = df['stoch_rsi_k'].fillna(50)
                df['stoch_rsi_d'] = df['stoch_rsi_d'].fillna(50)
            except Exception as e:
                print(f"Warning: Error calculating momentum indicators: {str(e)}")
                df['rsi'] = 50
                df['stoch_rsi_k'] = 50
                df['stoch_rsi_d'] = 50
            
            # 4. Volatility indicators
            for period in self.lookback_periods:
                try:
                    bb = BollingerBands(close=df['close'], window=period)
                    df[f'bb_upper_{period}'] = bb.bollinger_hband()
                    df[f'bb_lower_{period}'] = bb.bollinger_lband()
                    
                    df[f'bb_upper_{period}'] = df[f'bb_upper_{period}'].fillna(df['close'])
                    df[f'bb_lower_{period}'] = df[f'bb_lower_{period}'].fillna(df['close'])
                    df[f'bb_width_{period}'] = ((df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df['close']).fillna(0)
                except Exception as e:
                    print(f"Warning: Error calculating Bollinger Bands for period {period}: {str(e)}")
                    df[f'bb_upper_{period}'] = df['close']
                    df[f'bb_lower_{period}'] = df['close']
                    df[f'bb_width_{period}'] = 0
            
            try:
                atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
                df['atr'] = atr.average_true_range()
                df['atr'] = df['atr'].fillna(0)
            except Exception as e:
                print(f"Warning: Error calculating ATR: {str(e)}")
                df['atr'] = 0
            
            # 5. Volume indicators
            try:
                mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
                df['mfi'] = mfi.money_flow_index()
                df['mfi'] = df['mfi'].fillna(50)
                
                obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
                df['obv'] = obv.on_balance_volume()
                df['obv'] = df['obv'].ffill().fillna(0)
            except Exception as e:
                print(f"Warning: Error calculating volume indicators: {str(e)}")
                df['mfi'] = 50
                df['obv'] = 0
            
            # Final cleanup
            df = df.ffill().bfill()
            
            # Verify all indicators are clean
            if df.isnull().any().any():
                missing_cols = df.columns[df.isnull().any()].tolist()
                raise ValueError(f"Indicators contain NaN values after calculation in columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            raise

    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add machine learning based features."""
        try:
            # Verify required columns exist
            required_columns = ['close', 'volume', 'adx', 'rsi']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Market regime detection
            regime_features = df[required_columns].copy()
            
            # Forward fill and then backward fill to handle any remaining NaNs
            regime_features = regime_features.ffill().bfill()
            
            # Replace any infinite values with NaN and then fill them
            regime_features = regime_features.replace([np.inf, -np.inf], np.nan)
            regime_features = regime_features.ffill().bfill()
            
            # Verify no NaN values remain
            if regime_features.isnull().any().any():
                missing_cols = regime_features.columns[regime_features.isnull().any()].tolist()
                raise ValueError(f"NaN values still present after cleaning in columns: {missing_cols}")
            
            # Scale features for regime detection
            regime_features_scaled = self.scaler.fit_transform(regime_features)
            
            # Detect market regimes
            df['market_regime'] = pd.Series(
                self.regime_classifier.fit_predict(regime_features_scaled),
                index=df.index
            )
            
            # Anomaly detection
            df['is_anomaly'] = pd.Series(
                self.anomaly_detector.fit_predict(regime_features_scaled),
                index=df.index
            )
            
            # Feature interactions (with error checking)
            df['trend_strength'] = df['adx'].mul(df['di_plus'].sub(df['di_minus'])).fillna(0)
            df['volume_price_trend'] = df['obv'].mul(df['rsi']).fillna(0)
            
            # Volatility regime (with error checking)
            for period in self.lookback_periods:
                vol_data = df[f'volatility_{period}'].rolling(period).mean()
                vol_data = vol_data.fillna(method='bfill').fillna(0)  # Handle edge cases
                try:
                    df[f'volatility_regime_{period}'] = pd.Series(
                        pd.qcut(vol_data, 4, labels=False, duplicates='drop'),
                        index=df.index
                    ).fillna(0)  # Fill any remaining NaN with 0 (neutral regime)
                except Exception as e:
                    print(f"Warning: Could not calculate volatility regime for period {period}: {str(e)}")
                    df[f'volatility_regime_{period}'] = 0  # Default to neutral regime
            
            # Final validation
            if df.isnull().any().any():
                missing_cols = df.columns[df.isnull().any()].tolist()
                raise ValueError(f"NaN values found after ML feature engineering in columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            print(f"Error adding ML features: {str(e)}")
            raise
    
    def _clean_and_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and scale features."""
        try:
            # Forward fill missing values
            df = df.ffill()
            
            # Remove remaining NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error cleaning features: {str(e)}")
            raise


class MachineLearningStrategy(BaseStrategy):
    """Advanced trading strategy with ML-based feature engineering."""
    
    def __init__(self, **kwargs):
        params = {
            'lookback_periods': kwargs.get('lookback_periods', [5, 10, 20, 50, 100]),
            'min_samples': kwargs.get('min_samples', 100),
            'signal_threshold': kwargs.get('signal_threshold', 0.7),
            'stop_loss_atr': kwargs.get('stop_loss_atr', 2.0),
            'take_profit_atr': kwargs.get('take_profit_atr', 3.0),
            'max_holding_period': kwargs.get('max_holding_period', 20),
            'trend_filter_adx': kwargs.get('trend_filter_adx', 25),
            'volatility_filter_percentile': kwargs.get('volatility_filter_percentile', 80)
        }
        super().__init__(params)
        self.feature_engineer = FeatureEngineer(params['lookback_periods'])
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical and ML-based indicators."""
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Sort index to ensure proper calculation of indicators
            df = df.sort_index()
            
            # Calculate technical indicators first
            df = self.feature_engineer._add_technical_indicators(df)
            
            # Then calculate ML features using the technical indicators
            df = self.feature_engineer._add_ml_features(df)
            
            # Final validation
            if df.isnull().any().any():
                missing_cols = df.columns[df.isnull().any()].tolist()
                raise ValueError(f"NaN values found in columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            raise

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using ML features."""
        try:
            # First calculate all indicators
            df = self.calculate_indicators(data.copy())
            
            if 'adx' not in df.columns:
                raise ValueError("Technical indicators not properly calculated")
            
            # Generate entry signals
            df['long_signal'] = self._generate_long_signal(df)
            df['short_signal'] = self._generate_short_signal(df)
            
            # Generate exit signals
            df['exit_signal'] = self._generate_exit_signal(df)
            
            # Fill any NaN values
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            raise
    
    def _generate_long_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate long entry signals."""
        # Trend filter
        trend_filter = (df['adx'] > self.params['trend_filter_adx']) & (df['di_plus'] > df['di_minus'])
        
        # Volatility filter
        volatility = df['atr'] / df['close']
        vol_filter = volatility > volatility.quantile(self.params['volatility_filter_percentile'] / 100)
        
        # ML-based filters
        regime_filter = df['market_regime'].isin([0, 1])  # Bullish regimes
        anomaly_filter = df['is_anomaly'] == 1  # Not an anomaly
        
        # Momentum conditions
        momentum_filter = (df['rsi'] > 50) & (df['mfi'] > 50)
        
        # Combine all filters
        return (trend_filter & vol_filter & regime_filter & anomaly_filter & momentum_filter).astype(int)
    
    def _generate_short_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate short entry signals."""
        # Trend filter
        trend_filter = (df['adx'] > self.params['trend_filter_adx']) & (df['di_minus'] > df['di_plus'])
        
        # Volatility filter
        volatility = df['atr'] / df['close']
        vol_filter = volatility > volatility.quantile(self.params['volatility_filter_percentile'] / 100)
        
        # ML-based filters
        regime_filter = df['market_regime'].isin([2, 3])  # Bearish regimes
        anomaly_filter = df['is_anomaly'] == 1  # Not an anomaly
        
        # Momentum conditions
        momentum_filter = (df['rsi'] < 50) & (df['mfi'] < 50)
        
        # Combine all filters
        return (trend_filter & vol_filter & regime_filter & anomaly_filter & momentum_filter).astype(int)
    
    def _generate_exit_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate exit signals."""
        # Stop loss and take profit using ATR
        long_stop = df['close'] < (df['close'].shift(1) - self.params['stop_loss_atr'] * df['atr'])
        long_profit = df['close'] > (df['close'].shift(1) + self.params['take_profit_atr'] * df['atr'])
        
        short_stop = df['close'] > (df['close'].shift(1) + self.params['stop_loss_atr'] * df['atr'])
        short_profit = df['close'] < (df['close'].shift(1) - self.params['take_profit_atr'] * df['atr'])
        
        # Time-based exit
        holding_period = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['long_signal'].iloc[i-1] == 1 or df['short_signal'].iloc[i-1] == 1:
                holding_period.iloc[i] = holding_period.iloc[i-1] + 1
            else:
                holding_period.iloc[i] = 0
                
        time_exit = holding_period >= self.params['max_holding_period']
        
        # Regime change exit
        regime_change = df['market_regime'] != df['market_regime'].shift(1)
        
        # Combine exit signals
        return ((long_stop | long_profit | short_stop | short_profit | time_exit | regime_change) & (holding_period > 0)).astype(int)
    
    def calculate_position_size(self, 
                              equity: float,
                              entry_price: float,
                              atr: float,
                              volatility_ratio: float = 1.0,
                              market_impact_score: float = 1.0) -> float:
        """Calculate position size with ML-based adjustments."""
        try:
            # Base position size using ATR for risk management
            risk_per_trade = 0.02  # 2% risk per trade
            risk_amount = equity * risk_per_trade
            
            # Stop loss in absolute terms
            stop_loss = self.params['stop_loss_atr'] * atr
            
            # Calculate base position size
            base_position_size = risk_amount / stop_loss
            
            # Adjust position size based on volatility and market impact
            adjusted_position_size = base_position_size * volatility_ratio * market_impact_score
            
            # Apply maximum position size limit
            max_position_size = equity * 0.1  # Maximum 10% of equity per trade
            final_position_size = min(adjusted_position_size, max_position_size)
            
            return final_position_size
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return 0.0
