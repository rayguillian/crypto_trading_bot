from binance.client import Client
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

class BinanceDataFetcher:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Binance data fetcher. Can be used with or without API keys.
        Without keys, only public endpoints are available.
        """
        self.client = Client(api_key, api_secret) if api_key and api_secret else Client()
        
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_historical_klines(self, 
                            symbol: str,
                            interval: str,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical kline data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_time: Start time for data fetch
            end_time: End time for data fetch
            limit: Maximum number of klines to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to milliseconds timestamp if provided
            start_str = int(start_time.timestamp() * 1000) if start_time else None
            end_str = int(end_time.timestamp() * 1000) if end_time else None
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching kline data: {e}")
            raise
            
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logging.error(f"Error fetching current price: {e}")
            raise
            
    def get_exchange_info(self, symbol: str) -> Dict:
        """Get exchange information for a symbol."""
        try:
            info = self.client.get_exchange_info()
            symbol_info = next(
                (item for item in info['symbols'] if item['symbol'] == symbol),
                None
            )
            
            if not symbol_info:
                raise ValueError(f"Symbol {symbol} not found")
                
            # Extract relevant trading rules
            filters = {f['filterType']: f for f in symbol_info['filters']}
            
            return {
                'min_qty': float(filters['LOT_SIZE']['minQty']),
                'max_qty': float(filters['LOT_SIZE']['maxQty']),
                'step_size': float(filters['LOT_SIZE']['stepSize']),
                'min_notional': float(filters['MIN_NOTIONAL']['minNotional'])
            }
            
        except Exception as e:
            logging.error(f"Error fetching exchange info: {e}")
            raise
