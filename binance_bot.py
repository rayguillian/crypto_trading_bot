import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import pickle
from pathlib import Path
import logging
from typing import Dict, Optional
import os
from dotenv import load_dotenv

class BinanceBot:
    """Live trading bot for Binance using optimized strategies."""
    
    def __init__(self, config_path: str = "best_strategy_config.yaml"):
        """
        Initialize Binance trading bot.
        
        Args:
            config_path: Path to strategy configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        
        # Load strategy
        self.strategy = self.load_strategy()
        
        # Trading parameters
        self.symbol = self.config['symbol']
        self.timeframe = self.config['timeframe']
        self.position = None
        self.last_signal = 0
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_strategy(self):
        """Load the trading strategy."""
        try:
            with open(self.config['strategy_path'], 'rb') as f:
                strategy = pickle.load(f)
            
            # Set strategy parameters
            strategy.set_parameters(self.config['parameters'])
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error loading strategy: {e}")
            raise
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch recent market data."""
        try:
            # Calculate periods needed based on strategy lookback
            max_lookback = max(self.strategy.params.get('lookback_periods', [100]))
            periods = max_lookback + 100  # Add buffer
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.timeframe,
                limit=periods
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_position(self) -> Optional[Dict]:
        """Get current position information."""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for position in positions:
                if position['symbol'] == self.symbol:
                    return position
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching position: {e}")
            return None
    
    def execute_trade(self, signal: int, data: pd.DataFrame):
        """Execute trade based on signal."""
        try:
            if signal == 0:
                return
                
            position = self.get_position()
            current_position = float(position['contracts']) if position else 0
            
            # Calculate position size and risk parameters
            equity = float(self.exchange.fetch_balance()['total']['USDT'])
            entry_price = data['close'].iloc[-1]
            atr = data['atr'].iloc[-1]  # Assuming ATR is calculated in strategy
            volatility_ratio = data['volatility_20'].iloc[-1] / data['volatility_50'].iloc[-1]
            
            position_size, stop_loss, take_profit = self.strategy.calculate_position_size(
                equity=equity,
                entry_price=entry_price,
                atr=atr,
                volatility_ratio=volatility_ratio,
                market_impact_score=0.1  # Conservative estimate
            )
            
            # Execute orders
            if signal > 0 and current_position <= 0:  # Long entry
                if current_position < 0:  # Close short first
                    self.exchange.create_market_buy_order(
                        self.symbol,
                        abs(current_position)
                    )
                
                # Enter long position
                order = self.exchange.create_market_buy_order(
                    self.symbol,
                    position_size
                )
                
                # Set stop loss and take profit
                self.exchange.create_order(
                    self.symbol,
                    'stop',
                    'sell',
                    position_size,
                    stop_loss,
                    {'stopPrice': stop_loss}
                )
                
                self.exchange.create_order(
                    self.symbol,
                    'take_profit',
                    'sell',
                    position_size,
                    take_profit,
                    {'stopPrice': take_profit}
                )
                
                self.logger.info(f"Entered long position: {order}")
                
            elif signal < 0 and current_position >= 0:  # Short entry
                if current_position > 0:  # Close long first
                    self.exchange.create_market_sell_order(
                        self.symbol,
                        current_position
                    )
                
                # Enter short position
                order = self.exchange.create_market_sell_order(
                    self.symbol,
                    position_size
                )
                
                # Set stop loss and take profit
                self.exchange.create_order(
                    self.symbol,
                    'stop',
                    'buy',
                    position_size,
                    stop_loss,
                    {'stopPrice': stop_loss}
                )
                
                self.exchange.create_order(
                    self.symbol,
                    'take_profit',
                    'buy',
                    position_size,
                    take_profit,
                    {'stopPrice': take_profit}
                )
                
                self.logger.info(f"Entered short position: {order}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def run(self):
        """Run the trading bot."""
        self.logger.info("Starting trading bot...")
        
        while True:
            try:
                # Fetch latest data
                data = self.fetch_data()
                if len(data) == 0:
                    continue
                
                # Generate trading signals
                signals = self.strategy.generate_signals(data)
                current_signal = signals['entry'].iloc[-1]
                
                # Execute trade if signal changes
                if current_signal != self.last_signal:
                    self.execute_trade(current_signal, data)
                    self.last_signal = current_signal
                
                # Log current state
                position = self.get_position()
                if position:
                    self.logger.info(
                        f"Current position: {position['contracts']} contracts, "
                        f"PnL: {position['unrealizedPnl']}"
                    )
                
                # Wait for next candle
                timeframe_minutes = self.exchange.parse_timeframe(self.timeframe)
                time.sleep(timeframe_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    bot = BinanceBot()
    bot.run()
