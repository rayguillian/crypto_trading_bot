import argparse
import logging
from datetime import datetime, timedelta
from core.strategy_evaluator import StrategyEvaluator
from strategies import get_strategy_class
from binance.client import Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Test a trading strategy')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy class name')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (e.g., 1h, 4h, 1d)')
    parser.add_argument('--period', type=str, default='7d', help='Test period (e.g., 7d, 30d)')
    args = parser.parse_args()

    # Initialize Binance client
    client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

    # Get strategy class
    strategy_class = get_strategy_class(args.strategy)
    if not strategy_class:
        logger.error(f"Strategy {args.strategy} not found")
        return

    # Parse period
    period_value = int(args.period[:-1])
    period_unit = args.period[-1].lower()
    if period_unit == 'd':
        end_time = datetime.now()
        start_time = end_time - timedelta(days=period_value)
    else:
        logger.error("Unsupported period unit. Use 'd' for days")
        return

    # Initialize strategy evaluator
    evaluator = StrategyEvaluator(
        strategy_class=strategy_class,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_time=start_time,
        end_time=end_time
    )

    # Run backtest
    try:
        results = evaluator.run_backtest()
        
        # Print results
        print("\nBacktest Results:")
        print(f"Total Returns: {results['total_returns']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}")

if __name__ == '__main__':
    main()
