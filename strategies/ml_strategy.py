import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from .base_strategy import BaseStrategy
from core.signal import Signal

class MLStrategy(BaseStrategy):
    """Machine Learning based trading strategy using RandomForest."""
    
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.lookback_period = 20
        self.prediction_threshold = 0.001  # 0.1% price movement threshold
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for ML model."""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'rolling_vol_{window}'] = df['returns'].rolling(window=window).std()
            
        # Technical indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        
        return df.dropna()
        
    def prepare_ml_data(self, data: pd.DataFrame, target_horizon: int = 5) -> tuple:
        """Prepare data for ML model."""
        features_df = self.generate_features(data)
        
        # Create target variable (future returns)
        features_df['target'] = features_df['close'].pct_change(target_horizon).shift(-target_horizon)
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        # Select feature columns
        feature_columns = [col for col in features_df.columns if col not in 
                         ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = features_df[feature_columns]
        y = features_df['target']
        
        return X, y
        
    def train_model(self, data: pd.DataFrame):
        """Train the ML model."""
        X, y = self.prepare_ml_data(data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.trained = True
        
    def predict_returns(self, features: pd.DataFrame) -> float:
        """Predict future returns using trained model."""
        if not self.trained:
            return 0.0
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        return self.model.predict(features_scaled)[0]
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using ML predictions."""
        if len(data) < self.lookback_period:
            return pd.DataFrame(index=data.index)
            
        # Train model if not trained
        if not self.trained:
            self.train_model(data)
            
        # Generate features for prediction
        features_df = self.generate_features(data)
        feature_columns = [col for col in features_df.columns if col not in 
                         ['open', 'high', 'low', 'close', 'volume']]
                         
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = 0
        signals['exit'] = 0
        
        for i in range(self.lookback_period, len(data)):
            # Get features for current timestamp
            current_features = features_df.iloc[i:i+1][feature_columns]
            
            # Make prediction
            predicted_return = self.predict_returns(current_features)
            
            # Generate signals based on predictions
            if predicted_return > self.prediction_threshold:
                signals.iloc[i]['entry'] = 1  # Long signal
            elif predicted_return < -self.prediction_threshold:
                signals.iloc[i]['entry'] = -1  # Short signal
            elif abs(predicted_return) < self.prediction_threshold * 0.5:
                signals.iloc[i]['exit'] = 1  # Exit signal
                
        return signals
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_macd(self, prices: pd.Series, 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> tuple:
        """Calculate MACD technical indicator."""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal
        
    def set_parameters(self, params: Dict):
        """Set strategy parameters."""
        if 'lookback_period' in params:
            self.lookback_period = params['lookback_period']
        if 'prediction_threshold' in params:
            self.prediction_threshold = params['prediction_threshold']
        if 'n_estimators' in params:
            self.model = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params.get('max_depth', 10),
                random_state=42
            )
            self.trained = False
