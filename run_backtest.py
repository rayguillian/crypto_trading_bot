import logging
from datetime import datetime, timedelta
from strategies.moving_average_strategy import MovingAverageStrategy
from core.strategy_evaluator import StrategyEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/backtest_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Define evaluation period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)  # 1 month of data
    
    # Trading pair and timeframe
    symbol = 'BTCUSDT'
    timeframe = '1h'
    
    # Create and run evaluator
    evaluator = StrategyEvaluator(
        strategy_class=MovingAverageStrategy,
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        save_dir="strategy_results",
        min_sharpe=1.0,  # More lenient requirements for testing
        min_profit_factor=1.2,
        max_drawdown=0.25
    )
    
    try:
        results = evaluator.evaluate_strategy()
        logger.info(f"Backtest Results: {results}")
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
