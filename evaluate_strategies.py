import logging
from datetime import datetime, timedelta
from pathlib import Path
from strategies.moving_average_strategy import MovingAverageStrategy
from strategies.rsi_macd_strategy import RSIMACDStrategy
from strategies.ma_crossover_strategy import MACrossoverStrategy
from core.strategy_evaluator import StrategyEvaluator
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/strategy_evaluation_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_strategy(historical_data: pd.DataFrame, strategy_type: str) -> dict:
    """
    Evaluate a trading strategy on historical data
    
    Args:
        historical_data: DataFrame with OHLCV data
        strategy_type: Type of strategy to evaluate ('trend_following', 'mean_reversion', 'momentum')
        
    Returns:
        dict with strategy performance metrics
    """
    if strategy_type == 'trend_following':
        signals = trend_following_strategy(historical_data)
    elif strategy_type == 'mean_reversion':
        signals = mean_reversion_strategy(historical_data)
    elif strategy_type == 'momentum':
        signals = momentum_strategy(historical_data)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Calculate returns and metrics
    returns = calculate_returns(historical_data, signals)
    metrics = calculate_metrics(returns)
    
    return metrics

def trend_following_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Simple trend following strategy using EMA crossover
    """
    # Calculate EMAs
    ema_short = talib.EMA(df['close'], timeperiod=20)
    ema_long = talib.EMA(df['close'], timeperiod=50)
    
    # Generate signals
    signals = pd.Series(index=df.index, data=0)
    signals[ema_short > ema_long] = 1  # Long signal
    signals[ema_short < ema_long] = -1  # Short signal
    
    return signals

def mean_reversion_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Mean reversion strategy using RSI
    """
    # Calculate RSI
    rsi = talib.RSI(df['close'], timeperiod=14)
    
    # Generate signals
    signals = pd.Series(index=df.index, data=0)
    signals[rsi < 30] = 1  # Oversold -> Long signal
    signals[rsi > 70] = -1  # Overbought -> Short signal
    
    return signals

def momentum_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Momentum strategy using MACD and ADX
    """
    # Calculate MACD
    macd, signal, _ = talib.MACD(df['close'])
    
    # Calculate ADX
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Generate signals
    signals = pd.Series(index=df.index, data=0)
    
    # Only take signals when ADX indicates strong trend
    strong_trend = adx > 25
    signals[(macd > signal) & strong_trend] = 1  # Long signal
    signals[(macd < signal) & strong_trend] = -1  # Short signal
    
    return signals

def calculate_returns(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """
    Calculate strategy returns based on signals
    """
    # Calculate price returns
    price_returns = df['close'].pct_change()
    
    # Strategy returns (assuming we can short)
    strategy_returns = signals.shift(1) * price_returns
    
    # Remove NA values
    strategy_returns = strategy_returns.fillna(0)
    
    return strategy_returns

def calculate_metrics(returns: pd.Series) -> dict:
    """
    Calculate strategy performance metrics
    """
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    daily_returns = returns[returns != 0]  # Only consider days with positions
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    # Calculate drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Win rate
    win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns)
    
    return {
        'total_return': total_return,
        'total_returns': total_return,  
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

def get_historical_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a symbol
    """
    # Convert timeframe to minutes for the API
    timeframe_minutes = {
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }[timeframe]
    
    # TODO: Implement actual data fetching from exchange
    # For now, return dummy data
    periods = int((end_date - start_date).total_seconds() / (60 * timeframe_minutes))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start=start_date, end=end_date, periods=periods),
        'open': np.random.normal(100, 1, periods).cumsum(),
        'high': np.random.normal(100, 1, periods).cumsum() + 1,
        'low': np.random.normal(100, 1, periods).cumsum() - 1,
        'close': np.random.normal(100, 1, periods).cumsum(),
        'volume': np.random.normal(1000000, 100000, periods)
    })
    
    df.set_index('timestamp', inplace=True)
    return df

def main():
    # Define evaluation period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=60)  # 2 months of data
    
    # Trading pairs to test (formatted correctly for Binance)
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['1h', '4h']
    
    strategies = [
        MovingAverageStrategy,
        RSIMACDStrategy,
        MACrossoverStrategy
    ]
    
    for symbol in symbols:
        for timeframe in timeframes:
            for strategy_class in strategies:
                logger.info(f"Evaluating {strategy_class.__name__} on {symbol} {timeframe}")
                
                evaluator = StrategyEvaluator(
                    strategy_class=strategy_class,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    save_dir="strategy_results",
                    min_sharpe=1.5,
                    min_profit_factor=1.5,
                    max_drawdown=0.2
                )
                
                try:
                    # Run backtest
                    results = evaluator.run_backtest()
                    
                    logger.info(f"Results for {strategy_class.__name__} - {symbol} - {timeframe}:")
                    logger.info(f"Performance metrics: {results}")
                    logger.info("-" * 80)
                    
                except Exception as e:
                    logger.error(f"Error evaluating {strategy_class.__name__} on {symbol} {timeframe}: {str(e)}")
                    continue
                
                time.sleep(5)  # Add a 5-second delay between evaluations

if __name__ == "__main__":
    main()
