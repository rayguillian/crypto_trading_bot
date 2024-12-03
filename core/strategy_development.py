from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import ta
from datetime import datetime, timedelta
import logging
from models.database import Strategy, db
import json

logger = logging.getLogger(__name__)

class StrategyDevelopment:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        
    def fetch_historical_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """Fetch historical klines data from Binance"""
        try:
            # Calculate start time
            start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            
            # Fetch klines
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=str(start_time),
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df.set_index('timestamp')
            
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch historical data: {str(e)}")
            raise
            
    def calculate_indicators(self, df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        """Calculate technical indicators based on strategy type"""
        try:
            if strategy_type == 'Momentum':
                # RSI
                df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
                # MACD
                macd = ta.trend.MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                # Stochastic
                stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch.stoch()
                df['stoch_d'] = stoch.stoch_signal()
                
            elif strategy_type == 'Mean Reversion':
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df['close'])
                df['bb_upper'] = bollinger.bollinger_hband()
                df['bb_middle'] = bollinger.bollinger_mavg()
                df['bb_lower'] = bollinger.bollinger_lband()
                # ATR
                df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
                
            elif strategy_type == 'Breakout':
                # Donchian Channels
                df['donchian_high'] = df['high'].rolling(window=20).max()
                df['donchian_low'] = df['low'].rolling(window=20).min()
                # Volume Indicators
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                
            elif strategy_type == 'Machine Learning':
                # Feature Engineering for ML
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(window=20).std()
                df['momentum'] = df['returns'].rolling(window=10).mean()
                df['trend'] = df['close'].rolling(window=50).mean()
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {str(e)}")
            raise
            
    def backtest_strategy(self, df: pd.DataFrame, strategy_type: str, params: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Backtest strategy with given parameters"""
        try:
            signals = pd.DataFrame(index=df.index)
            signals['position'] = 0
            
            if strategy_type == 'Momentum':
                # RSI + MACD Strategy
                rsi_oversold = params.get('rsi_oversold', 30)
                rsi_overbought = params.get('rsi_overbought', 70)
                
                signals.loc[(df['rsi'] < rsi_oversold) & (df['macd'] > df['macd_signal']), 'position'] = 1
                signals.loc[(df['rsi'] > rsi_overbought) & (df['macd'] < df['macd_signal']), 'position'] = -1
                
            elif strategy_type == 'Mean Reversion':
                # Bollinger Bands Strategy
                signals.loc[df['close'] < df['bb_lower'], 'position'] = 1
                signals.loc[df['close'] > df['bb_upper'], 'position'] = -1
                
            elif strategy_type == 'Breakout':
                # Donchian Channel Breakout
                signals.loc[df['close'] > df['donchian_high'].shift(1), 'position'] = 1
                signals.loc[df['close'] < df['donchian_low'].shift(1), 'position'] = -1
                
            # Calculate returns
            signals['returns'] = df['close'].pct_change()
            signals['strategy_returns'] = signals['position'].shift(1) * signals['returns']
            
            metrics = self._calculate_metrics(signals)
            
            return signals, metrics
            
        except Exception as e:
            logger.error(f"Failed to backtest strategy: {str(e)}")
            raise
            
    def _calculate_metrics(self, signals):
        """Calculate strategy performance metrics"""
        try:
            # Calculate returns
            total_return = (signals['strategy_returns'].sum() * 100)  # Convert to percentage
            
            # Calculate Sharpe ratio with error handling
            returns_std = signals['strategy_returns'].std()
            if returns_std > 0:  # Only calculate if std dev is positive
                sharpe_ratio = np.sqrt(252) * signals['strategy_returns'].mean() / returns_std
            else:
                sharpe_ratio = 0
            
            # Calculate win rate
            winning_trades = len(signals[signals['strategy_returns'] > 0])
            total_trades = len(signals[signals['strategy_returns'] != 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + signals['strategy_returns']).cumprod()
            rolling_max = cumulative_returns.expanding(min_periods=1).max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() * 100  # Convert to percentage
            
            # Calculate profit factor
            gross_profits = signals[signals['strategy_returns'] > 0]['strategy_returns'].sum()
            gross_losses = abs(signals[signals['strategy_returns'] < 0]['strategy_returns'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else 0
            
            # Calculate Calmar ratio
            if max_drawdown != 0:
                calmar_ratio = abs(total_return / max_drawdown)
            else:
                calmar_ratio = 0
            
            return {
                'total_return': float(total_return),
                'total_returns': float(total_return),  # Added for backward compatibility
                'sharpe_ratio': float(sharpe_ratio),
                'win_rate': float(win_rate),
                'max_drawdown': float(max_drawdown),
                'profit_factor': float(profit_factor),
                'calmar_ratio': float(calmar_ratio),
                'total_trades': int(total_trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'total_return': 0,
                'total_returns': 0,  # Added for backward compatibility
                'sharpe_ratio': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'calmar_ratio': 0,
                'total_trades': 0
            }
            
    def optimize_parameters(self, df: pd.DataFrame, strategy_type: str) -> Dict:
        """Optimize strategy parameters"""
        try:
            best_metrics = {'sharpe_ratio': -np.inf}
            best_params = {}
            
            if strategy_type == 'Momentum':
                for rsi_oversold in range(20, 41, 5):
                    for rsi_overbought in range(60, 81, 5):
                        params = {
                            'rsi_oversold': rsi_oversold,
                            'rsi_overbought': rsi_overbought
                        }
                        _, metrics = self.backtest_strategy(df, strategy_type, params)
                        
                        if metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']:
                            best_metrics = metrics
                            best_params = params
                            
            elif strategy_type == 'Mean Reversion':
                for bb_period in range(10, 31, 5):
                    for bb_std in [1.5, 2.0, 2.5]:
                        params = {
                            'bb_period': bb_period,
                            'bb_std': bb_std
                        }
                        _, metrics = self.backtest_strategy(df, strategy_type, params)
                        
                        if metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']:
                            best_metrics = metrics
                            best_params = params
                            
            return {'params': best_params, 'metrics': best_metrics}
            
        except Exception as e:
            logger.error(f"Failed to optimize parameters: {str(e)}")
            raise
            
    def save_strategy(self, name: str, strategy_type: str, symbol: str, params: Dict, metrics: Dict) -> Strategy:
        """Save strategy to database"""
        try:
            strategy = Strategy(
                name=name,
                type=strategy_type,
                symbol=symbol,
                parameters=json.dumps(params),
                backtesting_results=json.dumps(metrics),
                is_active=False,
                created_at=datetime.now()
            )
            
            db.session.add(strategy)
            db.session.commit()
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to save strategy: {str(e)}")
            db.session.rollback()
            raise
            
    def generate_signals(self, symbol: str, timeframe: str, strategy_type: str) -> pd.DataFrame:
        """Generate trading signals based on strategy type"""
        try:
            # Get historical data
            data = self.fetch_historical_data(symbol, timeframe, 30)
            if data is None or len(data) == 0:
                return None
                
            # Calculate indicators based on strategy type
            if strategy_type == 'Momentum':
                signals = self._momentum_strategy(data)
            elif strategy_type == 'Mean Reversion':
                signals = self._mean_reversion_strategy(data)
            elif strategy_type == 'Breakout':
                signals = self._breakout_strategy(data)
            else:
                logging.error(f"Unknown strategy type: {strategy_type}")
                return None
                
            # Add required fields for trading executor
            signals['symbol'] = symbol
            signals['price'] = data['close']
            signals['timestamp'] = data.index
            
            return signals
            
        except Exception as e:
            logging.error(f"Error generating signals: {str(e)}")
            return None

    def _momentum_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum strategy signals"""
        try:
            # Calculate moving averages
            data['SMA20'] = data['close'].rolling(window=20).mean()
            data['SMA50'] = data['close'].rolling(window=50).mean()
            
            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals['action'] = 'HOLD'
            
            # Buy when short MA crosses above long MA
            signals.loc[(data['SMA20'] > data['SMA50']) & 
                       (data['SMA20'].shift(1) <= data['SMA50'].shift(1)), 'action'] = 'BUY'
            
            # Sell when short MA crosses below long MA
            signals.loc[(data['SMA20'] < data['SMA50']) & 
                       (data['SMA20'].shift(1) >= data['SMA50'].shift(1)), 'action'] = 'SELL'
            
            return signals
            
        except Exception as e:
            logging.error(f"Error generating momentum signals: {str(e)}")
            return None

    def _mean_reversion_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion strategy signals"""
        try:
            # Calculate Bollinger Bands
            data['SMA20'] = data['close'].rolling(window=20).mean()
            data['STD20'] = data['close'].rolling(window=20).std()
            data['UpperBand'] = data['SMA20'] + (data['STD20'] * 2)
            data['LowerBand'] = data['SMA20'] - (data['STD20'] * 2)
            
            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals['action'] = 'HOLD'
            
            # Buy when price crosses below lower band
            signals.loc[data['close'] < data['LowerBand'], 'action'] = 'BUY'
            
            # Sell when price crosses above upper band
            signals.loc[data['close'] > data['UpperBand'], 'action'] = 'SELL'
            
            return signals
            
        except Exception as e:
            logging.error(f"Error generating mean reversion signals: {str(e)}")
            return None

    def _breakout_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout strategy signals"""
        try:
            # Calculate support and resistance levels
            data['High20'] = data['high'].rolling(window=20).max()
            data['Low20'] = data['low'].rolling(window=20).min()
            
            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals['action'] = 'HOLD'
            
            # Buy on resistance breakout
            signals.loc[data['close'] > data['High20'].shift(1), 'action'] = 'BUY'
            
            # Sell on support breakdown
            signals.loc[data['close'] < data['Low20'].shift(1), 'action'] = 'SELL'
            
            return signals
            
        except Exception as e:
            logging.error(f"Error generating breakout signals: {str(e)}")
            return None
            
    def develop_strategy(self, name, strategy_type, symbol, interval='1h', lookback_days=30):
        """Develop and test a new trading strategy"""
        try:
            # Get historical data
            historical_data = self.fetch_historical_data(symbol, interval, lookback_days)
            
            # Generate trading signals based on strategy type
            signals = self.calculate_indicators(historical_data, strategy_type)
            
            # Calculate strategy metrics
            metrics = self.optimize_parameters(signals, strategy_type)
            
            # Save strategy to database
            strategy = Strategy(
                name=name,
                strategy_type=strategy_type,  
                symbol=symbol,
                timeframe=interval,
                description=f"Auto-generated {strategy_type} strategy for {symbol}",
                backtesting_results=json.dumps(metrics['metrics']),
                total_return=metrics['metrics'].get('total_returns', 0),
                sharpe_ratio=metrics['metrics'].get('sharpe_ratio', 0),
                win_rate=metrics['metrics'].get('win_rate', 0),
                max_drawdown=metrics['metrics'].get('max_drawdown', 0),
                profit_factor=0,
                calmar_ratio=0
            )
            
            db.session.add(strategy)
            db.session.commit()
            
            return {
                'strategy_id': strategy.id,
                'metrics': metrics['metrics']
            }
            
        except Exception as e:
            logger.error(f"Strategy development failed: {str(e)}")
            raise
