import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, api_key=None, api_secret=None):
        self.client = Client(api_key, api_secret) if api_key and api_secret else Client()
    
    def get_historical_data(self, symbol, interval, start_str, end_str=None):
        """Get historical klines/candlestick data"""
        klines = self.client.get_historical_klines(
            symbol, interval, start_str, end_str
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'buy_base_volume',
            'buy_quote_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df