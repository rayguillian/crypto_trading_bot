# Consolidated Trading Bot Structure

import pandas as pd
import numpy as np
from ta.momentum import KAMAIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any
from datetime import datetime, timedelta
from binance.client import Client
import os
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
import yaml

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='crypto_trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class DataLoader:
    def __init__(self, api_key: str = '', api_secret: str = ''):
        """
        Initialize with Binance API credentials.
        """
        self.client = Client(api_key, api_secret)

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = '1h') -> pd.DataFrame:
        """
        Fetch historical data from Binance.
        """
        logging.info(f"Fetching historical data for {symbol}")
        try:
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            klines = self.client.get_historical_klines(symbol, interval, start_str=str(start_ts), end_str=str(end_ts))
            if not klines:
                raise ValueError("No data received from Binance")
            df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            df.sort_values('time', inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise

class Strategy:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Initialize strategy with data and initial capital.
        """
        self.data = data
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position = None
        self.entry_price = 0.0
        self.equity_curve = [initial_capital]

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on indicators.
        """
        kama = KAMAIndicator(close=self.data['close'], window=14, pow1=2, pow2=30, fillna=True)
        self.data['kama'] = kama.kama()
        adx = ADXIndicator(high=self.data['high'], low=self.data['low'], close=self.data['close'], window=14, fillna=True)
        self.data['adx'] = adx.adx()
        signals = pd.DataFrame(index=self.data.index)
        signals['entry'] = np.where(self.data['close'] > self.data['kama'], 1, -1)
        signals['exit'] = np.where(self.data['adx'] < 25, 1, 0)
        return signals

class Backtester:
    def __init__(self, strategy: Strategy):
        """
        Initialize backtester with a strategy.
        """
        self.strategy = strategy

    def run_backtest(self) -> Dict[str, Any]:
        """
        Run backtest and calculate performance metrics.
        """
        logging.info("Starting backtest")
        signals = self.strategy.generate_signals()
        for i, row in self.strategy.data.iterrows():
            if self.strategy.position is None and signals.at[i, 'entry'] == 1:
                self.strategy.position = 'long'
                self.strategy.entry_price = row['close']
                logging.info(f"Entering long position at {row['close']}")
            elif self.strategy.position == 'long' and signals.at[i, 'exit'] == 1:
                self.strategy.balance += row['close'] - self.strategy.entry_price
                self.strategy.equity_curve.append(self.strategy.balance)
                logging.info(f"Exiting long position at {row['close']}")
                self.strategy.position = None
        return {
            'final_balance': self.strategy.balance,
            'equity_curve': self.strategy.equity_curve
        }

    def plot_equity_curve(self):
        """
        Plot the equity curve.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.strategy.equity_curve, label='Equity Curve')
        plt.title('Equity Curve')
        plt.xlabel('Trades')
        plt.ylabel('Balance')
        plt.legend()
        plt.grid(True)
        plt.show()

class TradingSystem:
    """Main trading system integrating all components."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize trading system with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.backtester = Backtester(Strategy(pd.DataFrame()))
        
        # Initialize logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config['logging']['file']
        )
        
        self.logger = logging.getLogger(__name__)

    def run_complete_pipeline(self):
        """Run the complete trading pipeline."""
        results = {}
        
        for symbol in self.config['strategy']['symbols']:
            self.logger.info(f"Running pipeline for {symbol}")
            
            try:
                # Run backtest directly without optimization
                backtest_results = self.run_backtest(
                    strategy_name=self.config['strategy']['default'],
                    timeframe=self.config['strategy']['timeframe'],
                    trading_pair=symbol
                )
                
                if backtest_results:
                    results[symbol] = backtest_results
                    self.logger.info(f"Successfully completed pipeline for {symbol}")
                else:
                    self.logger.warning(f"No results generated for {symbol}")
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        return results

    def find_successful_strategy(self):
        # Define parameters
        strategies = ['basic', 'advanced', 'machine_learning']  # Example strategy names
        timeframes = ['1h', '4h', '1d']
        trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

        # Define success criteria
        success_criteria = {
            'min_win_rate': 50.0,  # Minimum win rate in percentage
            'min_profit_factor': 1.5,
            'min_total_return': 10.0  # Minimum total return in percentage
        }

        # Iterate over combinations
        for strategy_name in strategies:
            for timeframe in timeframes:
                for trading_pair in trading_pairs:
                    self.logger.info(f"Testing strategy: {strategy_name}, Timeframe: {timeframe}, Trading Pair: {trading_pair}")

                    # Run backtest
                    backtest_results = self.run_backtest(strategy_name, timeframe, trading_pair)

                    # Evaluate results
                    if (backtest_results['win_rate'] >= success_criteria['min_win_rate'] and
                        backtest_results['profit_factor'] >= success_criteria['min_profit_factor'] and
                        backtest_results['total_return'] >= success_criteria['min_total_return']):
                        
                        self.logger.info(f"Successful strategy found: {strategy_name}, Timeframe: {timeframe}, Trading Pair: {trading_pair}")
                        # Save successful strategy
                        self.save_strategy(strategy_name, timeframe, trading_pair, backtest_results)
                        return

        self.logger.info("No successful strategy found.")

    def run_backtest(self, strategy_name, timeframe, trading_pair):
        # Placeholder for running backtest
        # Implement the backtest logic here
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_return': 0.0
        }

    def save_strategy(self, strategy_name, timeframe, trading_pair, results):
        # Placeholder for saving strategy
        # Implement the logic to save the strategy details
        pass

app = Flask(__name__)
CORS(app)

@app.route('/api/strategies')
def get_strategies():
    """Return list of available strategies and their current status."""
    try:
        logger.info("Fetching strategies")
        strategies = [
            {
                'id': 'rsi_macd',
                'name': 'RSI MACD Strategy',
                'description': 'Combines RSI and MACD indicators for trading signals',
                'status': 'active'
            },
            {
                'id': 'machine_learning',
                'name': 'Machine Learning Strategy',
                'description': 'Uses ML models for price prediction',
                'status': 'active'
            }
        ]
        return jsonify(strategies)
    except Exception as e:
        logger.error(f"Error in get_strategies: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<strategy_id>')
def get_strategy_details(strategy_id):
    """Return detailed information about a specific strategy."""
    try:
        logger.info(f"Fetching details for strategy {strategy_id}")
        
        # Mock data for demonstration
        strategies_details = {
            'rsi_macd': {
                'id': 'rsi_macd',
                'name': 'RSI MACD Strategy',
                'description': 'Combines RSI and MACD indicators for trading signals',
                'status': 'active',
                'performance': {
                    'total_return': 0.156,  # 15.6%
                    'win_rate': 0.68,       # 68%
                    'sharpe_ratio': 1.85,
                    'max_drawdown': 0.12,   # 12%
                    'total_trades': 142
                },
                'parameters': {
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9
                }
            },
            'machine_learning': {
                'id': 'machine_learning',
                'name': 'Machine Learning Strategy',
                'description': 'Uses ML models for price prediction',
                'status': 'active',
                'performance': {
                    'total_return': 0.223,  # 22.3%
                    'win_rate': 0.72,       # 72%
                    'sharpe_ratio': 2.1,
                    'max_drawdown': 0.15,   # 15%
                    'total_trades': 98
                },
                'parameters': {
                    'model_type': 'LSTM',
                    'lookback_period': 30,
                    'feature_count': 15,
                    'training_epochs': 100,
                    'batch_size': 32,
                    'prediction_threshold': 0.65
                }
            }
        }
        
        if strategy_id not in strategies_details:
            return jsonify({'error': 'Strategy not found'}), 404
            
        return jsonify(strategies_details[strategy_id])
    except Exception as e:
        logger.error(f"Error fetching strategy details: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Endpoint to retrieve logs."""
    try:
        with open('logs/optimization_20241130.log', 'r') as log_file:
            logs = log_file.read()
        return jsonify({'logs': logs}), 200
    except FileNotFoundError:
        return jsonify({'error': 'Log file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(500)
def handle_500_error(e):
    logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def handle_404_error(e):
    logger.error(f"Not found error: {str(e)}")
    return jsonify({'error': 'Resource not found'}), 404

if __name__ == "__main__":
    try:
        # Initialize trading system
        print("Initializing Trading System...")
        try:
            trading_system = TradingSystem()
        except Exception as init_error:
            print(f"Failed to initialize TradingSystem: {init_error}")
            raise
        print("Trading System Initialized.")
        
        # Find successful strategy
        trading_system.find_successful_strategy()
        
        # Run the complete pipeline
        results = trading_system.run_complete_pipeline()
        
        # Print summary
        print("\nStrategy Optimization and Evaluation Summary:")
        print("=" * 50)
        
        for symbol, result in results.items():
            print(f"\n{symbol}:")
            print("-" * 30)
            print(f"Total Trades: {result['evaluation']['total_trades']}")
            print(f"Win Rate: {result['evaluation']['win_rate']:.2%}")
            print(f"Profit Factor: {result['evaluation']['profit_factor']:.2f}")
            print(f"Max Drawdown: {result['evaluation']['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {result['evaluation']['sharpe_ratio']:.2f}")
            print(f"Total Return: {result['evaluation']['total_return']:.2%}")
            print(f"Meets Criteria: {result['meets_criteria']}")
            print(f"Ready for Paper Trading: {result['ready_for_paper_trading']}")
            
            if result['meets_criteria']:
                print("\nOptimized Parameters:")
                for param, value in result['optimization_results']['optimized_params'].items():
                    print(f"{param}: {value}")
                    
        # Run Flask app
        print("Starting Flask server...")
        trading_system.logger.info("Starting Flask server on port 5001")
        app.run(port=5001, debug=True)
        print("Flask server started.")
    except Exception as e:
        trading_system.logger.error(f"Failed to start server: {str(e)}", exc_info=True)
