#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import json
from binance.client import Client
from concurrent.futures import ThreadPoolExecutor
from itertools import product

from core.data_fetcher import BinanceDataFetcher
from strategies.machine_learning_strategy import MachineLearningStrategy
from core.strategy_evaluator import StrategyEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/strategy_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyTester:
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize the strategy tester with optional API credentials."""
        self.client = Client(api_key, api_secret) if api_key and api_secret else Client()
        self.data_fetcher = BinanceDataFetcher(self.client)
        self.evaluator = StrategyEvaluator()
        
    def fetch_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch historical data from Binance."""
        try:
            data = self.data_fetcher.fetch_historical_data(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date or datetime.now().strftime('%Y-%m-%d')
            )
            logger.info(f"Fetched {len(data)} data points for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def generate_parameter_combinations(self) -> List[Dict]:
        """Generate different parameter combinations to test."""
        param_grid = {
            'lookback_periods': [[5, 10, 20], [10, 20, 30], [20, 30, 50]],
            'regime_threshold': [0.1, 0.2, 0.3],
            'volatility_lookback': [10, 20, 30],
            'momentum_threshold': [0.02, 0.03, 0.05],
            'risk_tolerance': [0.02, 0.03, 0.05],
            'position_size': [0.1, 0.2, 0.3]
        }
        
        combinations = []
        for values in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), values))
            combinations.append(param_dict)
        
        return combinations

    def evaluate_strategy(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Evaluate a strategy with given parameters."""
        try:
            strategy = MachineLearningStrategy()
            strategy.set_parameters(**params)
            
            # Split data into training and testing sets
            train_size = int(len(data) * 0.7)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Train the strategy
            strategy.train(train_data)
            
            # Generate signals on test data
            signals = strategy.generate_signals(test_data)
            
            # Calculate performance metrics
            metrics = self.evaluator.evaluate_strategy(
                test_data,
                signals,
                initial_capital=10000,
                transaction_cost=0.001
            )
            
            return {
                'parameters': params,
                'metrics': metrics,
                'train_size': len(train_data),
                'test_size': len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {str(e)}")
            return None

    def test_strategies(self, symbol: str, interval: str, start_date: str, end_date: str = None) -> List[Dict]:
        """Test multiple strategy configurations and return results."""
        try:
            # Fetch historical data
            data = self.fetch_historical_data(symbol, interval, start_date, end_date)
            
            # Generate parameter combinations
            param_combinations = self.generate_parameter_combinations()
            logger.info(f"Testing {len(param_combinations)} parameter combinations")
            
            # Evaluate strategies in parallel
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_params = {
                    executor.submit(self.evaluate_strategy, data, params): params 
                    for params in param_combinations
                }
                
                for future in future_to_params:
                    result = future.result()
                    if result:
                        results.append(result)
            
            # Sort results by Sharpe ratio
            results.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
            
            # Log results
            self.log_results(results, symbol, interval)
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing strategies: {str(e)}")
            raise

    def log_results(self, results: List[Dict], symbol: str, interval: str):
        """Log strategy testing results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f'logs/strategy_results_{symbol}_{interval}_{timestamp}.json'
        
        summary = {
            'symbol': symbol,
            'interval': interval,
            'test_timestamp': timestamp,
            'total_strategies_tested': len(results),
            'best_strategies': results[:5]  # Top 5 strategies
        }
        
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Strategy testing results saved to {result_file}")
        
        # Log best strategy details
        best_strategy = results[0]
        logger.info("\nBest Strategy Results:")
        logger.info(f"Parameters: {best_strategy['parameters']}")
        logger.info(f"Metrics:")
        for metric, value in best_strategy['metrics'].items():
            logger.info(f"  {metric}: {value}")

def main():
    # Initialize tester
    tester = StrategyTester()
    
    # Test parameters
    symbol = 'BTCUSDT'
    interval = '1h'
    start_date = '2023-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Starting strategy testing for {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    try:
        results = tester.test_strategies(symbol, interval, start_date, end_date)
        logger.info("Strategy testing completed successfully")
        
    except Exception as e:
        logger.error(f"Strategy testing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
