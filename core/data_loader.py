import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv

class DataLoader:
    """Handle all data loading and preprocessing operations."""
    
    def __init__(self):
        """Initialize the data loader with Binance client."""
        load_dotenv()
        
        self.client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET')
        )
        
        # Cache for storing fetched data
        self._cache: Dict[str, pd.DataFrame] = {}
        
    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical klines data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1h', '4h', '1d')
            start_time: Start time in UTC format
            end_time: End time in UTC format
            limit: Maximum number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None if error occurs
        """
        cache_key = f"{symbol}_{interval}_{start_time}_{end_time}"
        
        # Return cached data if available
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            # Convert times to milliseconds timestamp if provided
            start_ts = int(pd.Timestamp(start_time).timestamp() * 1000) if start_time else None
            end_ts = int(pd.Timestamp(end_time).timestamp() * 1000) if end_time else None
            
            # Fetch klines from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_ts,
                end_str=end_ts,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Calculate additional features
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['atr'] = self._calculate_atr(df)
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            # Store in cache
            self._cache[cache_key] = df
            
            return df[['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility', 'atr', 'volume_ma']]
            
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return None
            
    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current market data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Dictionary with current market data or None if error occurs
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return {
                'price': float(ticker['price']),
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            logging.error(f"Error fetching live data for {symbol}: {e}")
            return None
            
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Current price as float or None if error occurs
        """
        live_data = self.get_live_data(symbol)
        return live_data['price'] if live_data else None
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
        
    def get_exchange_info(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol exchange information.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Dictionary with symbol information or None if error occurs
        """
        try:
            info = self.client.get_symbol_info(symbol)
            return {
                'base_asset': info['baseAsset'],
                'quote_asset': info['quoteAsset'],
                'min_qty': float(info['filters'][2]['minQty']),
                'step_size': float(info['filters'][2]['stepSize']),
                'tick_size': float(info['filters'][0]['tickSize']),
                'min_notional': float(info['filters'][3]['minNotional'])
            }
            
        except Exception as e:
            logging.error(f"Error fetching exchange info for {symbol}: {e}")
            return None
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw data by handling missing values and normalizing features.
        
        Args:
            df: Raw DataFrame with OHLCV data
            
        Returns:
            Preprocessed DataFrame
        """
        # Create copy to avoid modifying original
        data = df.copy()
        
        # First validate the raw data
        data = self.validate_data(data)
        
        # Handle missing values
        data = self.handle_missing_data(data)
        
        # Normalize price data
        data = self.normalize_data(data)
        
        return data
        
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and integrity.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        
        Raises:
            ValueError: If data validation fails
        """
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check for invalid values
        if df[required_cols].isnull().any().any():
            raise ValueError("Found NaN values in required columns")
            
        # Verify price relationships
        invalid_prices = (
            (df['high'] < df['low']) |
            (df['close'] < df['low']) |
            (df['close'] > df['high']) |
            (df['open'] < df['low']) |
            (df['open'] > df['high'])
        )
        if invalid_prices.any():
            raise ValueError("Invalid price relationships detected")
            
        # Verify positive volumes
        if (df['volume'] < 0).any():
            raise ValueError("Negative volumes detected")
            
        return df
        
    def resample_data(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            df: DataFrame to resample
            interval: Target interval (e.g., '1h', '4h', '1d')
            
        Returns:
            Resampled DataFrame
        """
        # Ensure DataFrame has DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or timestamp column")
        
        resampled = df.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Calculate technical indicators
        resampled['returns'] = resampled['close'].pct_change()
        resampled['volatility'] = resampled['returns'].rolling(window=20).std()
        resampled['atr'] = self._calculate_atr(resampled)
        resampled['volume_ma'] = resampled['volume'].rolling(window=20).mean()
        
        return resampled
        
    def merge_data_sources(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple data sources into a single DataFrame.
        
        Args:
            dfs: List of DataFrames to merge
            
        Returns:
            Merged DataFrame
        """
        if not dfs:
            raise ValueError("No DataFrames provided for merging")
            
        # Start with first DataFrame
        result = dfs[0].copy()
        
        # Merge remaining DataFrames
        for i, df in enumerate(dfs[1:], 1):
            result = pd.merge(
                result,
                df,
                left_index=True,
                right_index=True,
                how='outer',
                suffixes=('', f'_{i}')
            )
            
        return result
        
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data through interpolation or forward filling.
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with handled missing values
        """
        data = df.copy()
        
        # Forward fill missing values for OHLCV data
        price_cols = ['open', 'high', 'low', 'close']
        if any(col in data.columns for col in price_cols):
            data[price_cols] = data[price_cols].ffill()
        
        # Fill volume with 0
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        
        return data
        
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features to a common scale.
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            Normalized DataFrame
        """
        data = df.copy()
        
        # Normalize prices relative to first close price
        if 'close' in data.columns:
            first_price = data['close'].iloc[0]
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    # Create new normalized columns with _norm suffix
                    data[f'{col}_norm'] = data[col] / first_price - 1
                    # Update original columns
                    data[col] = data[f'{col}_norm']
        
        # Min-max normalization for volume
        if 'volume' in data.columns:
            vol_min = data['volume'].min()
            vol_max = data['volume'].max()
            if vol_max > vol_min:
                data['volume_norm'] = (data['volume'] - vol_min) / (vol_max - vol_min)
                data['volume'] = data['volume_norm']
            else:
                data['volume_norm'] = 0
                data['volume'] = 0
            
        return data
