import os
from dotenv import load_dotenv
from core.strategy_optimizer import StrategyOptimizer
import logging
import json
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)

def save_optimization_results(results, output_dir='optimization_results'):
    """Save optimization results to files."""
    if not results:
        print("No results to save")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert results to DataFrame for analysis
    results_data = []
    for result in results:
        if not isinstance(result, dict):  # Handle OptimizationResult objects
            result = {
                'strategy_name': result.strategy_name,
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'parameters': result.parameters,
                'metrics': result.metrics,
                'risk_metrics': result.risk_metrics
            }
            
        data = {
            'strategy_name': result['strategy_name'],
            'symbol': result['symbol'],
            'timeframe': result['timeframe'],
            'total_return': result['metrics']['total_return'],
            'max_drawdown': result['metrics']['max_drawdown'],
            'sharpe_ratio': result['metrics']['sharpe_ratio'],
            'win_rate': result['metrics']['win_rate'],
            'profit_factor': result['metrics'].get('profit_factor', 0),
            'calmar_ratio': result['risk_metrics']['calmar_ratio'],
            'avg_trade_duration': result['risk_metrics']['avg_trade_duration'],
            'max_consecutive_losses': result['risk_metrics']['max_consecutive_losses'],
            'risk_adjusted_return': result['risk_metrics']['risk_adjusted_return'],
            'meets_criteria': result['metrics'].get('meets_criteria', False)
        }
        data.update({f'param_{k}': v for k, v in result['parameters'].items()})
        results_data.append(data)
    
    df = pd.DataFrame(results_data)
    
    # Save detailed CSV report
    csv_file = f'{output_dir}/optimization_results_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    # Save summary report
    summary_file = f'{output_dir}/optimization_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write("Strategy Optimization Results Summary\n")
        f.write("===================================\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Total Strategies Tested: {len(df)}\n")
        f.write(f"Strategies Meeting Criteria: {df['meets_criteria'].sum()}\n")
        f.write(f"Success Rate: {(df['meets_criteria'].sum() / len(df)):.2%}\n\n")
        
        # Statistics for all strategies
        f.write("All Strategies Performance:\n")
        f.write("-------------------------\n")
        f.write(f"Average Return: {df['total_return'].mean():.2f}%\n")
        f.write(f"Best Return: {df['total_return'].max():.2f}%\n")
        f.write(f"Worst Return: {df['total_return'].min():.2f}%\n")
        f.write(f"Average Sharpe: {df['sharpe_ratio'].mean():.2f}\n")
        f.write(f"Average Win Rate: {df['win_rate'].mean():.2%}\n")
        f.write(f"Average Max Drawdown: {df['max_drawdown'].mean():.2f}%\n\n")
        
        # Statistics for passing strategies
        passing_df = df[df['meets_criteria']]
        if len(passing_df) > 0:
            f.write("Passing Strategies Performance:\n")
            f.write("-----------------------------\n")
            f.write(f"Average Return: {passing_df['total_return'].mean():.2f}%\n")
            f.write(f"Best Return: {passing_df['total_return'].max():.2f}%\n")
            f.write(f"Average Sharpe: {passing_df['sharpe_ratio'].mean():.2f}\n")
            f.write(f"Average Win Rate: {passing_df['win_rate'].mean():.2%}\n")
            f.write(f"Average Max Drawdown: {passing_df['max_drawdown'].mean():.2f}%\n\n")
            
            # Top performing strategies
            f.write("Top 5 Strategies by Risk-Adjusted Return:\n")
            top_return = passing_df.nlargest(5, 'risk_adjusted_return')
            for _, row in top_return.iterrows():
                f.write(f"\n{row['strategy_name']} on {row['symbol']} {row['timeframe']}:\n")
                f.write(f"  Total Return: {row['total_return']:.2f}%\n")
                f.write(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}\n")
                f.write(f"  Win Rate: {row['win_rate']:.2%}\n")
                f.write(f"  Max Drawdown: {row['max_drawdown']:.2f}%\n")
                f.write(f"  Parameters: {', '.join(f'{k.replace('param_', '')}={v}' for k, v in row.items() if k.startswith('param_'))}\n")
        else:
            f.write("No strategies met the minimum performance criteria.\n\n")
        
        # Strategy-specific analysis
        f.write("\nStrategy-specific Performance:\n")
        for strategy in df['strategy_name'].unique():
            strategy_df = df[df['strategy_name'] == strategy]
            passing_strategy = strategy_df[strategy_df['meets_criteria']]
            f.write(f"\n{strategy}:\n")
            f.write(f"Total Tested: {len(strategy_df)}\n")
            f.write(f"Passed Criteria: {len(passing_strategy)}\n")
            f.write(f"Average Return: {strategy_df['total_return'].mean():.2f}%\n")
            f.write(f"Best Return: {strategy_df['total_return'].max():.2f}%\n")
            f.write(f"Worst Return: {strategy_df['total_return'].min():.2f}%\n")
            f.write(f"Average Sharpe: {strategy_df['sharpe_ratio'].mean():.2f}\n")
            f.write(f"Average Win Rate: {strategy_df['win_rate'].mean():.2%}\n")
            
        # Trading pair analysis
        f.write("\nTrading Pair Performance:\n")
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            passing_symbol = symbol_df[symbol_df['meets_criteria']]
            f.write(f"\n{symbol}:\n")
            f.write(f"Total Tested: {len(symbol_df)}\n")
            f.write(f"Passed Criteria: {len(passing_symbol)}\n")
            f.write(f"Average Return: {symbol_df['total_return'].mean():.2f}%\n")
            f.write(f"Best Return: {symbol_df['total_return'].max():.2f}%\n")
            f.write(f"Best Strategy: {symbol_df.loc[symbol_df['total_return'].idxmax(), 'strategy_name']}\n")
            
        # Timeframe analysis
        f.write("\nTimeframe Performance:\n")
        for timeframe in df['timeframe'].unique():
            timeframe_df = df[df['timeframe'] == timeframe]
            passing_timeframe = timeframe_df[timeframe_df['meets_criteria']]
            f.write(f"\n{timeframe}:\n")
            f.write(f"Total Tested: {len(timeframe_df)}\n")
            f.write(f"Passed Criteria: {len(passing_timeframe)}\n")
            f.write(f"Average Return: {timeframe_df['total_return'].mean():.2f}%\n")
            f.write(f"Best Return: {timeframe_df['total_return'].max():.2f}%\n")
            f.write(f"Best Strategy: {timeframe_df.loc[timeframe_df['total_return'].idxmax(), 'strategy_name']}\n")
    
    print(f"\nOptimization results saved to:")
    print(f"- CSV file: {csv_file}")
    print(f"- Summary file: {summary_file}")
    
    return csv_file, summary_file

def camel_to_snake(name):
    """Convert CamelCase to snake_case."""
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def main():
    """Run strategy optimization."""
    load_dotenv()
    optimizer = StrategyOptimizer()
    
    # List of strategies to optimize
    strategies = [
        'TrendVolumeStrategy',
        'BollingerRSIStrategy',
        'MachineLearningStrategy',
        'MeanReversionStrategy',
        'RSIMACDStrategy',
        'MACrossoverStrategy',
        'BbandsVolumeStrategy',
        'KAMAStrategy',
        'MovingAverageStrategy'
    ]
    
    all_results = []
    
    for strategy_name in strategies:
        logging.info(f"\nOptimizing {strategy_name}...")
        
        try:
            # Convert strategy name to file name format
            module_name = camel_to_snake(strategy_name)
            
            # Get strategy class
            strategy_module = __import__(f'strategies.{module_name}', fromlist=[strategy_name])
            strategy_class = getattr(strategy_module, strategy_name)
            
            # Add strategy to optimizer's strategies
            optimizer.strategies[strategy_name] = strategy_class
            
            # Add default parameter ranges if not present
            if strategy_name not in optimizer.parameter_ranges:
                optimizer.parameter_ranges[strategy_name] = {
                    'window': [20, 30],
                    'threshold': [0.5, 0.7],
                    'stop_loss': [0.02, 0.03],
                    'take_profit': [0.03, 0.05]
                }
            
            # Run optimization for each trading pair and timeframe
            for symbol in optimizer.trading_pairs:
                for timeframe in optimizer.timeframes:
                    logging.info(f"\nTesting {strategy_name} on {symbol} {timeframe}")
                    
                    try:
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
                            
                            if result:
                                all_results.append(result)
                                
                                # Log promising results
                                metrics = result.metrics
                                if metrics.get('weighted_score', 0) > 0.7:  # Adjust threshold as needed
                                    logging.info(f"\nPromising configuration found:")
                                    logging.info(f"Strategy: {strategy_name}")
                                    logging.info(f"Symbol: {symbol}")
                                    logging.info(f"Timeframe: {timeframe}")
                                    logging.info(f"Parameters: {params}")
                                    logging.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
                                    logging.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                                    logging.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
                                    logging.info(f"Weighted Score: {metrics.get('weighted_score', 0):.2f}")
                    
                    except Exception as e:
                        logging.error(f"Error testing {strategy_name} with {symbol} {timeframe}: {str(e)}")
                        continue
        
        except Exception as e:
            logging.error(f"Error loading strategy {strategy_name}: {str(e)}")
            continue
    
    # Save results
    if all_results:
        save_optimization_results(all_results)
        logging.info("\nOptimization completed. Results saved to optimization_results directory.")
    else:
        logging.warning("\nNo valid results found during optimization.")

if __name__ == "__main__":
    main()
