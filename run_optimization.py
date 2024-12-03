import logging
from datetime import datetime, timedelta
from core.strategy_optimizer import StrategyOptimizer
from pathlib import Path
import json
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/optimization_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Initialize optimizer
    optimizer = StrategyOptimizer()
    
    # Define evaluation period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=60)  # 2 months of data
    
    # Define parameter ranges for optimization
    parameter_ranges = {
        'MovingAverageStrategy': {
            'short_window': list(range(5, 21, 5)),
            'long_window': list(range(20, 101, 20)),
            'rsi_period': list(range(7, 22, 7)),
            'rsi_overbought': list(range(65, 81, 5)),
            'rsi_oversold': list(range(20, 36, 5))
        },
        'RSIMACDStrategy': {
            'rsi_period': list(range(7, 22, 7)),
            'macd_fast': list(range(8, 21, 4)),
            'macd_slow': list(range(20, 41, 5)),
            'macd_signal': list(range(5, 16, 5)),
            'rsi_overbought': list(range(65, 81, 5)),
            'rsi_oversold': list(range(20, 36, 5))
        },
        'BollingerRSIStrategy': {
            'bb_period': list(range(10, 31, 5)),
            'bb_std': [1.5, 2.0, 2.5, 3.0],
            'rsi_period': list(range(7, 22, 7)),
            'rsi_overbought': list(range(65, 81, 5)),
            'rsi_oversold': list(range(20, 36, 5))
        }
    }
    
    # Update optimizer's parameter ranges
    optimizer.parameter_ranges.update(parameter_ranges)
    
    # Trading pairs to focus on
    focus_pairs = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['1h', '4h']
    
    all_results = []
    
    for symbol in focus_pairs:
        for timeframe in timeframes:
            logger.info(f"\nOptimizing strategies for {symbol} {timeframe}")
            
            try:
                # Test each strategy
                for strategy_name, param_range in parameter_ranges.items():
                    logger.info(f"\nTesting {strategy_name}")
                    
                    # Generate parameter combinations
                    param_combinations = optimizer.generate_parameter_combinations(strategy_name)
                    
                    # Test each parameter combination
                    for params in param_combinations:
                        result = optimizer.evaluate_strategy(
                            symbol=symbol,
                            timeframe=timeframe,
                            strategy_name=strategy_name,
                            parameters=params
                        )
                        
                        if result and result.metrics.get('weighted_score', 0) > 0.7:
                            all_results.append(result)
                            logger.info(f"\nPromising configuration found:")
                            logger.info(f"Strategy: {strategy_name}")
                            logger.info(f"Parameters: {params}")
                            logger.info(f"Total Return: {result.metrics.get('total_return', 0):.2%}")
                            logger.info(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
                            logger.info(f"Win Rate: {result.metrics.get('win_rate', 0):.2%}")
                            logger.info(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
            
            except Exception as e:
                logger.error(f"Error during optimization for {symbol} {timeframe}: {str(e)}", exc_info=True)
                continue
    
    # Save results
    if all_results:
        # Convert results to DataFrame
        results_data = []
        for result in all_results:
            data = {
                'strategy_name': result.strategy_name,
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'total_return': result.metrics['total_return'],
                'sharpe_ratio': result.metrics['sharpe_ratio'],
                'win_rate': result.metrics['win_rate'],
                'max_drawdown': result.metrics['max_drawdown'],
                'weighted_score': result.metrics['weighted_score']
            }
            data.update(result.parameters)
            results_data.append(data)
        
        df = pd.DataFrame(results_data)
        
        # Save to CSV
        output_dir = Path('optimization_results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(output_dir / f'optimization_results_{timestamp}.csv', index=False)
        
        # Generate summary report
        with open(output_dir / f'optimization_summary_{timestamp}.txt', 'w') as f:
            f.write("Strategy Optimization Results Summary\n")
            f.write("===================================\n\n")
            
            f.write("Top Performing Strategies:\n")
            f.write("------------------------\n")
            top_strategies = df.nlargest(5, 'weighted_score')
            for _, row in top_strategies.iterrows():
                f.write(f"\n{row['strategy_name']} on {row['symbol']} {row['timeframe']}:\n")
                f.write(f"  Total Return: {row['total_return']:.2%}\n")
                f.write(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}\n")
                f.write(f"  Win Rate: {row['win_rate']:.2%}\n")
                f.write(f"  Max Drawdown: {row['max_drawdown']:.2%}\n")
                f.write(f"  Weighted Score: {row['weighted_score']:.2f}\n")
                
                # Write parameters
                f.write("  Parameters:\n")
                for key, value in row.items():
                    if key not in ['strategy_name', 'symbol', 'timeframe', 'total_return', 
                                 'sharpe_ratio', 'win_rate', 'max_drawdown', 'weighted_score']:
                        f.write(f"    {key}: {value}\n")
        
        logger.info(f"\nOptimization completed. Results saved to {output_dir}")
    else:
        logger.warning("\nNo promising results found during optimization.")

if __name__ == "__main__":
    main()
