import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Type
import json
import os
from datetime import datetime
import pickle
from pathlib import Path
import yaml
from binance.client import Client
from core.backtester import Backtester
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy
import itertools
import logging

logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """Evaluates trading strategies and saves successful configurations."""
    
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 symbol: str,
                 timeframe: str,
                 start_time: datetime,
                 end_time: datetime,
                 save_dir: str = "strategy_results",
                 min_sharpe: float = 1.5,
                 min_profit_factor: float = 1.5,
                 max_drawdown: float = 0.2):
        """
        Initialize strategy evaluator.
        
        Args:
            strategy_class: Class of the strategy to evaluate
            symbol: Trading pair symbol
            timeframe: Timeframe for the data
            start_time: Start time for backtesting
            end_time: End time for backtesting
            save_dir: Directory to save strategy results
            min_sharpe: Minimum Sharpe ratio for strategy to be considered successful
            min_profit_factor: Minimum profit factor for strategy to be considered successful
            max_drawdown: Maximum allowable drawdown
        """
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        self.save_dir = Path(save_dir)
        self.min_sharpe = min_sharpe
        self.min_profit_factor = min_profit_factor
        self.max_drawdown = max_drawdown
        
        # Create necessary directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "strategies").mkdir(exist_ok=True)
        (self.save_dir / "results").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        
        # Initialize Binance client
        self.client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def get_historical_data(self) -> pd.DataFrame:
        """Fetch historical data from Binance"""
        klines = self.client.get_historical_klines(
            self.symbol,
            self.timeframe,
            self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            self.end_time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        df.set_index('timestamp', inplace=True)
        return df
        
    def run_backtest(self, initial_capital: float = 10000) -> Dict:
        """Run backtest with the strategy"""
        self.logger.info("Running backtest...")
        try:
            # Get historical data
            data = self.get_historical_data()
            self.logger.info("Historical data fetched successfully.")
            
            # Initialize strategy
            strategy = self.strategy_class()
            self.logger.info(f"Initialized strategy: {strategy.__class__.__name__}")
            
            # Run backtest
            backtester = Backtester(initial_capital=initial_capital)
            results = backtester.run_backtest(
                strategy=strategy,
                data=data,
                btc_price=data['close'].iloc[-1]
            )
            self.logger.info("Backtest completed successfully.")
            return {
                'total_returns': (results['portfolio_value'][-1] / initial_capital - 1) * 100,
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate'],
                'total_trades': results['total_trades']
            }
        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}")
            raise

    @staticmethod
    def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown from portfolio values"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd * 100
    
    def evaluate_strategy(self, data: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 10000, transaction_cost: float = 0.001) -> Dict:
        """
        Evaluate trading strategy performance with comprehensive metrics.
        
        Args:
            data: DataFrame with OHLCV data
            signals: DataFrame with trading signals
            initial_capital: Initial capital for trading
            transaction_cost: Transaction cost as a percentage
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Calculate returns
            position_sizes = signals['position_size'].fillna(0)
            price_returns = data['close'].pct_change()
            strategy_returns = position_sizes.shift(1) * price_returns
            strategy_returns = strategy_returns.fillna(0)
            
            # Calculate transaction costs
            position_changes = position_sizes.diff().abs()
            transaction_costs = position_changes * transaction_cost
            strategy_returns = strategy_returns - transaction_costs
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1
            
            # Calculate drawdowns
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Calculate Sharpe ratio (annualized)
            risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
            excess_returns = strategy_returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / strategy_returns.std()
            
            # Calculate Sortino ratio
            downside_returns = strategy_returns[strategy_returns < 0]
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
            
            # Calculate win rate and profit factor
            winning_trades = strategy_returns[strategy_returns > 0]
            losing_trades = strategy_returns[strategy_returns < 0]
            win_rate = len(winning_trades) / len(strategy_returns[strategy_returns != 0])
            profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else float('inf')
            
            # Calculate average trade metrics
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
            
            # Calculate exposure and turnover
            exposure = position_sizes.abs().mean()
            turnover = position_changes.sum() / 2  # Divide by 2 to count round-trip trades
            
            # Calculate recovery factor and risk-adjusted return
            recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else float('inf')
            risk_adjusted_return = total_return / (strategy_returns.std() * np.sqrt(252))
            
            # Calculate maximum consecutive wins and losses
            returns_binary = strategy_returns.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            consec_wins = max((sum(1 for _ in group) for value, group in itertools.groupby(returns_binary) if value == 1), default=0)
            consec_losses = max((sum(1 for _ in group) for value, group in itertools.groupby(returns_binary) if value == -1), default=0)
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'exposure': exposure,
                'turnover': turnover,
                'recovery_factor': recovery_factor,
                'risk_adjusted_return': risk_adjusted_return,
                'max_consecutive_wins': consec_wins,
                'max_consecutive_losses': consec_losses,
                'total_trades': len(strategy_returns[strategy_returns != 0]),
                'avg_trade_duration': None,  # TODO: Implement trade duration calculation
                'profit_per_trade': total_return / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {str(e)}")
            raise
    
    def _is_strategy_successful(self, metrics: Dict) -> bool:
        """Check if strategy meets success criteria."""
        return (
            metrics.get('sharpe_ratio', 0) >= self.min_sharpe and
            metrics.get('max_drawdown', 1) <= self.max_drawdown
        )
    
    def _save_strategy(self,
                      strategy: BaseStrategy,
                      results: Dict,
                      metrics: Dict,
                      params: Optional[Dict]) -> None:
        """Save successful strategy and its results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = f"strategy_{timestamp}"
        
        # Save strategy configuration
        config = {
            'strategy_class': strategy.__class__.__name__,
            'parameters': params or strategy.get_parameters(),
            'metrics': metrics
        }
        
        with open(self.save_dir / "strategies" / f"{strategy_name}_config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        # Save strategy object
        with open(self.save_dir / "strategies" / f"{strategy_name}.pkl", 'wb') as f:
            pickle.dump(strategy, f)
        
        # Save results
        pd.DataFrame(results).to_csv(self.save_dir / "results" / f"{strategy_name}_results.csv")
        
        # Generate and save plots
        self._save_strategy_plots(results, strategy_name)
        
        print(f"Successfully saved strategy: {strategy_name}")
        
    def _save_strategy_plots(self, results: Dict, strategy_name: str) -> None:
        """Generate and save strategy performance plots."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(rows=3, cols=1,
                              subplot_titles=('Equity Curve', 'Drawdown', 'Trade Returns'),
                              vertical_spacing=0.1)
            
            # Equity curve
            fig.add_trace(
                go.Scatter(y=results['portfolio_value'], name='Equity'),
                row=1, col=1
            )
            
            # Drawdown
            peak = np.maximum.accumulate(results['portfolio_value'])
            drawdown = (results['portfolio_value'] - peak) / peak
            fig.add_trace(
                go.Scatter(y=drawdown, name='Drawdown', fill='tozeroy'),
                row=2, col=1
            )
            
            # Trade returns
            fig.add_trace(
                go.Bar(y=results['trades']['profit'], name='Trade Returns'),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(height=1200, title_text=f"Strategy Performance: {strategy_name}")
            
            # Save plot
            fig.write_html(self.save_dir / "plots" / f"{strategy_name}_performance.html")
            
        except Exception as e:
            print(f"Error saving plots: {e}")
    
    def load_best_strategy(self, metric: str = 'sharpe_ratio') -> Optional[BaseStrategy]:
        """Load the best performing strategy based on specified metric."""
        try:
            best_metric = -np.inf
            best_strategy = None
            
            for config_file in (self.save_dir / "strategies").glob("*_config.yaml"):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if config['metrics'][metric] > best_metric:
                    best_metric = config['metrics'][metric]
                    strategy_file = config_file.stem.replace('_config', '') + '.pkl'
                    
                    with open(self.save_dir / "strategies" / strategy_file, 'rb') as f:
                        best_strategy = pickle.load(f)
            
            return best_strategy
            
        except Exception as e:
            print(f"Error loading best strategy: {e}")
            return None
    
    def generate_strategy_report(self) -> pd.DataFrame:
        """Generate a report comparing all saved strategies."""
        strategies = []
        
        for config_file in (self.save_dir / "strategies").glob("*_config.yaml"):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            strategy_data = {
                'strategy_name': config_file.stem.replace('_config', ''),
                'strategy_class': config['strategy_class'],
                **config['metrics']
            }
            strategies.append(strategy_data)
        
        return pd.DataFrame(strategies)
