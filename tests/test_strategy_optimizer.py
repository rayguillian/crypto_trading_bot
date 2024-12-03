import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy_optimizer import StrategyOptimizer
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def generate_mock_data(days: int = 100) -> pd.DataFrame:
    """Generate mock OHLCV data for testing."""
    # Add extra periods for indicator calculation
    lookback = 100  # Extra periods for technical indicators
    total_periods = (days + lookback) * 24  # Hourly data
    timestamps = pd.date_range(end=datetime.now(), periods=total_periods, freq='H')
    
    # Generate more realistic price movements
    np.random.seed(42)  # For reproducibility
    
    # Generate log returns with some autocorrelation
    returns = np.random.normal(0.0001, 0.002, total_periods)  # Daily returns
    # Add some autocorrelation
    returns = np.convolve(returns, np.ones(5)/5, mode='same')
    
    # Convert returns to prices
    base_price = 30000
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some seasonality
    seasonality = np.sin(np.linspace(0, 4*np.pi, total_periods)) * 100
    prices = prices + seasonality
    
    # Generate realistic OHLCV data
    data = pd.DataFrame(index=timestamps)
    
    # Generate open prices
    data['open'] = prices
    
    # Generate more realistic high/low prices
    volatility = np.abs(returns) * prices * 2
    data['high'] = prices + volatility
    data['low'] = prices - volatility
    
    # Ensure high is always highest and low is always lowest
    data['high'] = np.maximum(data['high'], data['open'])
    data['low'] = np.minimum(data['low'], data['open'])
    
    # Generate close prices between high and low
    weights = np.random.uniform(0, 1, total_periods)
    data['close'] = data['low'] + weights * (data['high'] - data['low'])
    
    # Generate volume with price-volume correlation
    base_volume = np.random.lognormal(10, 1, total_periods)
    volume_trend = np.abs(returns) * 10  # Higher volume on larger price moves
    data['volume'] = base_volume * (1 + volume_trend) * 1000
    
    # Add some gaps and jumps to simulate real market behavior
    jump_points = np.random.choice(total_periods, size=int(total_periods*0.01), replace=False)
    data.loc[data.index[jump_points], 'close'] *= np.random.uniform(0.95, 1.05, len(jump_points))
    
    # Ensure data types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Sort index and remove any duplicates
    data = data.sort_index().loc[~data.index.duplicated(keep='first')]
    
    # Remove the lookback period and return only the required days
    start_idx = len(data) - (days * 24)
    return data.iloc[start_idx:].copy()

def test_optimization():
    """Test strategy optimization capabilities."""
    optimizer = StrategyOptimizer()
    
    # Test parameters
    symbol = 'BTCUSDT'
    timeframe = '4h'
    strategy_name = 'MachineLearningStrategy'
    mock_btc_price = 35000.0
    
    # Mock the data fetching
    optimizer.data_fetcher.get_historical_klines = lambda *args, **kwargs: generate_mock_data()
    optimizer.data_fetcher.get_current_price = lambda *args: mock_btc_price
    
    # 1. Walk-Forward Analysis
    logging.info("Starting Walk-Forward Analysis...")
    wfa_results = optimizer.walk_forward_analysis(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        train_days=30,
        test_days=15,
        windows=4,
        btc_price=mock_btc_price
    )
    
    # Analyze WFA results
    if wfa_results:
        logging.info("\nWalk-Forward Analysis Summary:")
        total_returns = [r.metrics['total_return'] for r in wfa_results]
        sharpe_ratios = [r.metrics['sharpe_ratio'] for r in wfa_results]
        
        logging.info(f"Average Return across windows: {sum(total_returns)/len(total_returns):.2f}%")
        logging.info(f"Average Sharpe Ratio: {sum(sharpe_ratios)/len(sharpe_ratios):.2f}")
        
        # Find best window
        best_window = max(wfa_results, key=lambda x: x.metrics['total_return'])
        logging.info(f"\nBest Window Performance:")
        logging.info(f"Total Return: {best_window.metrics['total_return']:.2f}%")
        logging.info(f"Sharpe Ratio: {best_window.metrics['sharpe_ratio']:.2f}")
        logging.info(f"Parameters: {best_window.parameters}")
    
    # 2. Cross-Validation
    logging.info("\nStarting Cross-Validation...")
    
    # Use the best parameters from WFA for cross-validation
    if wfa_results:
        best_params = best_window.parameters
    else:
        # Default parameters if WFA didn't produce results
        best_params = {
            'lookback_periods': [5, 10, 20],
            'min_samples': 100,
            'signal_threshold': 0.7,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0,
            'max_holding_period': 20,
            'trend_filter_adx': 25,
            'volatility_filter_percentile': 80
        }
    
    cv_metrics = optimizer.cross_validate_strategy(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        parameters=best_params,
        folds=5,
        days_per_fold=15,
        btc_price=mock_btc_price
    )
    
    # Analyze CV results
    if cv_metrics:
        logging.info("\nCross-Validation Detailed Results:")
        for metric, values in cv_metrics.items():
            logging.info(f"\n{metric}:")
            logging.info(f"Mean: {values['mean']:.2f}")
            logging.info(f"Std: {values['std']:.2f}")
            logging.info(f"Min: {values['min']:.2f}")
            logging.info(f"Max: {values['max']:.2f}")

if __name__ == "__main__":
    test_optimization()
