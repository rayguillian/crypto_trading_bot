import argparse
import json
import os
from datetime import datetime, timedelta
from core.data import DataLoader
from strategies.moving_average import MovingAverageStrategy

def run_backtest(strategy_name, symbol='BTCUSDT', interval='1h', days=30):
    # Create data loader
    data_loader = DataLoader()
    
    # Get historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    data = data_loader.get_historical_data(
        symbol,
        interval,
        start_time.strftime('%Y-%m-%d')
    )
    
    # Initialize strategy
    if strategy_name == 'MA':
        strategy = MovingAverageStrategy()
    else:
        raise ValueError(f'Unknown strategy: {strategy_name}')
    
    # Run backtest
    results = strategy.backtest(data)
    
    # Save results
    results_dir = os.path.join('strategy_results', strategy_name)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f'{interval}_results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True, help='Strategy name')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', default='1h', help='Timeframe')
    parser.add_argument('--days', type=int, default=30, help='Backtest period in days')
    
    args = parser.parse_args()
    run_backtest(args.strategy, args.symbol, args.interval, args.days)