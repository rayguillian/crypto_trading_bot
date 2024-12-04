import json
import os

def load_backtest_results(strategy_name, timeframe):
    """Load backtest results for a given strategy and timeframe"""
    results_dir = os.path.join('strategy_results', strategy_name)
    results_file = os.path.join(results_dir, f'{timeframe}_results.json')
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'trades': [],
            'equity_curve': [],
            'metrics': {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
        }