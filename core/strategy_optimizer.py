import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from core.data_fetcher import BinanceDataFetcher
from core.backtester import Backtester
from core.risk_manager import RiskManager
from strategies.moving_average_strategy import MovingAverageStrategy
from strategies.bollinger_rsi_strategy import BollingerRSIStrategy
from strategies.trend_volume_strategy import TrendVolumeStrategy
from strategies.machine_learning_strategy import MachineLearningStrategy
import itertools

@dataclass
class OptimizationResult:
    strategy_name: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    risk_metrics: Dict[str, float]

class StrategyOptimizer:
    def __init__(self):
        """Initialize the Strategy Optimizer."""
        self.data_fetcher = BinanceDataFetcher()
        self.risk_manager = RiskManager()
        
        # Trading pairs focused on high-volume, liquid assets
        self.trading_pairs = [
            'BTCUSDT',  # Bitcoin
            'ETHUSDT',  # Ethereum
            'BNBUSDT',  # Binance Coin
            'SOLUSDT',  # Solana
            'MATICUSDT',  # Polygon
            'XRPUSDT',  # Ripple
            'ADAUSDT',  # Cardano
            'DOGEUSDT',  # Dogecoin
            'MANAUSDT',  # Decentraland
            'ALGOUSDT'  # Algorand
        ]
        
        # Timeframes focused on medium-term trends
        self.timeframes = ['1h', '4h', '1d']
        
        # Most reliable strategies
        self.strategies = {
            'BollingerRSIStrategy': BollingerRSIStrategy,
            'TrendVolumeStrategy': TrendVolumeStrategy,
            'MachineLearningStrategy': MachineLearningStrategy
        }
        
        # Optimized parameter ranges
        self.parameter_ranges = {
            'BollingerRSIStrategy': {
                'bb_window': [20, 30],
                'bb_std': [2.0, 2.5],
                'rsi_period': [14, 21],
                'rsi_overbought': [75, 80],
                'rsi_oversold': [20, 25]
            },
            'TrendVolumeStrategy': {
                'trend_period': [20, 30],
                'volume_ma_period': [20, 30],
                'price_change_threshold': [0.02, 0.03],
                'volume_multiplier': [2.0, 2.5]
            },
            'MachineLearningStrategy': {
                'lookback_period': [20, 30],
                'train_size': [0.7],
                'prediction_threshold': [0.65, 0.7],
                'n_estimators': [200],
                'max_depth': [10]
            }
        }
        
        # Performance weights for strategy evaluation
        self.performance_weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.20,
            'sortino_ratio': 0.15,
            'max_drawdown': 0.15,
            'win_rate': 0.10,
            'profit_factor': 0.10,
            'calmar_ratio': 0.05
        }
        
    def generate_parameter_combinations(self, strategy_name: str) -> List[Dict]:
        """Generate all possible parameter combinations for a strategy."""
        param_ranges = self.parameter_ranges[strategy_name]
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo)) for combo in combinations]
        
    def evaluate_strategy(self, 
                         symbol: str,
                         timeframe: str,
                         strategy_name: str,
                         parameters: Dict,
                         test_days: int = 60) -> OptimizationResult:
        """Evaluate a strategy with given parameters."""
        try:
            print(f"\nTesting {strategy_name} on {symbol} {timeframe}")
            print(f"Parameters: {parameters}")
            
            # Fetch historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=test_days)
            
            data = self.data_fetcher.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            # Initialize strategy and backtester
            strategy_class = self.strategies[strategy_name]
            strategy = strategy_class(params=parameters)
            backtester = Backtester(initial_capital=10000, commission=0.001)
            
            # Get current BTC price for volume calculations
            btc_price = self.data_fetcher.get_current_price('BTCUSDT')
            
            # Run backtest
            results = backtester.run_backtest(strategy, data, btc_price)
            
            # Calculate risk-adjusted metrics
            risk_metrics = {
                'calmar_ratio': results['total_return'] / results['max_drawdown'] if results['max_drawdown'] > 0 else float('inf'),
                'avg_trade_duration': self._calculate_avg_trade_duration(results['trades']),
                'max_consecutive_losses': self._calculate_max_consecutive_losses(results['trades']),
                'risk_adjusted_return': results['total_return'] / (results['max_drawdown'] + 1e-10)
            }

            # Create result object
            optimization_result = OptimizationResult(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                parameters=parameters,
                metrics=results,
                risk_metrics=risk_metrics
            )
            
            # Check if strategy meets criteria
            meets_criteria = self._meets_criteria(optimization_result)
            
            # Print performance metrics
            print(f"\nPerformance Metrics for {strategy_name} on {symbol} {timeframe}:")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Win Rate: {results['win_rate']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print(f"Risk-Adjusted Return: {risk_metrics['risk_adjusted_return']:.2f}")
            
            # Highlight if strategy meets performance criteria
            if meets_criteria:
                print("\n🌟 WINNING STRATEGY FOUND! 🌟")
                print("This combination meets all performance criteria")
                print("-" * 50)
            
            # Add meets_criteria to the result
            optimization_result.metrics['meets_criteria'] = meets_criteria
            return optimization_result
            
        except Exception as e:
            print(f"Error evaluating strategy: {e}")
            return None
            
    def optimize_all_combinations(self, test_days: int = 60) -> List[OptimizationResult]:
        """Test all combinations of strategies, parameters, trading pairs and timeframes."""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for strategy_name in self.strategies:
                parameter_combinations = self.generate_parameter_combinations(strategy_name)
                
                for symbol, timeframe, params in itertools.product(
                    self.trading_pairs, self.timeframes, parameter_combinations):
                    
                    futures.append(
                        executor.submit(
                            self.evaluate_strategy,
                            symbol=symbol,
                            timeframe=timeframe,
                            strategy_name=strategy_name,
                            parameters=params,
                            test_days=test_days
                        )
                    )
            
            # Collect results
            for future in futures:
                result = future.result()
                if result and self._meets_criteria(result):
                    all_results.append(result)
        
        # Sort results by risk-adjusted return
        all_results.sort(key=lambda x: x.metrics['total_return'] / (x.metrics['max_drawdown'] + 1e-10), reverse=True)
        return all_results
    
    def _meets_criteria(self, result: OptimizationResult) -> bool:
        """Check if strategy meets minimum performance criteria."""
        metrics = result.metrics
        return (
            metrics['total_return'] > 5.0 and        # At least 5% return
            metrics['sharpe_ratio'] > 1.5 and        # Better risk-adjusted return
            metrics['win_rate'] > 0.55 and           # Win more than lose
            metrics['max_drawdown'] < 15.0 and       # Reasonable drawdown
            metrics['profit_factor'] > 1.5 and       # Good profit/loss ratio
            result.risk_metrics['calmar_ratio'] > 0.5 # Good return/risk ratio
        )
    
    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in hours."""
        if not trades:
            return 0
            
        durations = []
        for trade in trades:
            if not trade['entry_time'] or not trade['exit_time']:
                continue
                
            entry = pd.to_datetime(trade['entry_time'])
            exit = pd.to_datetime(trade['exit_time'])
            duration = (exit - entry).total_seconds() / 3600
            durations.append(duration)
            
        return sum(durations) / len(durations) if durations else 0
    
    def _calculate_max_consecutive_losses(self, trades: List[Dict]) -> int:
        """Calculate maximum consecutive losing trades."""
        if not trades:
            return 0
            
        current_streak = 0
        max_streak = 0
        
        for trade in trades:
            if trade['pnl'] < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak

    def calculate_sortino_ratio(self, data: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)."""
        returns = data['returns'].fillna(0)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0
            
        downside_std = np.sqrt(np.mean(downside_returns**2))
        if downside_std == 0:
            return 0
            
        return np.sqrt(252) * (returns.mean() - risk_free_rate/252) / downside_std
    
    def calculate_calmar_ratio(self, data: pd.DataFrame) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        total_return = self.calculate_total_return(data)
        max_dd = self.calculate_max_drawdown(data)
        
        if max_dd == 0:
            return 0
            
        return total_return / abs(max_dd)

    def walk_forward_analysis(self,
                            strategy_name: str,
                            symbol: str,
                            timeframe: str,
                            train_days: int = 60,
                            test_days: int = 30,
                            windows: int = 6,
                            btc_price: float = None) -> List[OptimizationResult]:
        """
        Perform walk-forward analysis to validate strategy robustness.
        
        Args:
            strategy_name: Name of strategy to test
            symbol: Trading pair symbol
            timeframe: Trading timeframe
            train_days: Number of days for training window
            test_days: Number of days for testing window
            windows: Number of walk-forward windows
            btc_price: Current BTC price for volume calculations
            
        Returns:
            List of optimization results for each window
        """
        results = []
        
        # Get BTC price if not provided
        if btc_price is None:
            btc_price = self.data_fetcher.get_current_price('BTCUSDT')
        
        # Calculate total days needed
        total_days = (train_days + test_days) * windows
        
        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=total_days)
        
        data = self.data_fetcher.get_historical_klines(
            symbol=symbol,
            interval=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Split data into windows
        for i in range(windows):
            print(f"\nProcessing walk-forward window {i+1}/{windows}")
            
            # Calculate window indices
            train_start = i * (train_days + test_days)
            train_end = train_start + train_days
            test_start = train_end
            test_end = test_start + test_days
            
            # Get training and testing data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Find optimal parameters on training data
            parameter_combinations = self.generate_parameter_combinations(strategy_name)
            best_params = None
            best_score = float('-inf')
            
            for params in parameter_combinations:
                strategy = self.strategies[strategy_name](params=params)
                backtester = Backtester(initial_capital=10000, commission=0.001)
                
                try:
                    # Backtest on training data
                    train_results = backtester.run_backtest(strategy, train_data, btc_price)
                    
                    # Calculate score using weighted metrics
                    score = (
                        self.performance_weights['total_return'] * train_results['total_return'] +
                        self.performance_weights['sharpe_ratio'] * train_results['sharpe_ratio'] +
                        self.performance_weights['sortino_ratio'] * train_results['sortino_ratio'] -
                        self.performance_weights['max_drawdown'] * train_results['max_drawdown'] +
                        self.performance_weights['win_rate'] * train_results['win_rate'] +
                        self.performance_weights['profit_factor'] * train_results['profit_factor'] +
                        self.performance_weights['calmar_ratio'] * (
                            train_results['total_return'] / (train_results['max_drawdown'] + 1e-10)
                        )
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception as e:
                    print(f"Error in training: {e}")
                    continue
            
            if best_params:
                # Test optimal parameters on test data
                try:
                    strategy = self.strategies[strategy_name](params=best_params)
                    backtester = Backtester(initial_capital=10000, commission=0.001)
                    test_results = backtester.run_backtest(strategy, test_data, btc_price)
                    
                    # Calculate risk metrics
                    risk_metrics = {
                        'calmar_ratio': test_results['total_return'] / test_results['max_drawdown'] if test_results['max_drawdown'] > 0 else float('inf'),
                        'avg_trade_duration': self._calculate_avg_trade_duration(test_results['trades']),
                        'max_consecutive_losses': self._calculate_max_consecutive_losses(test_results['trades']),
                        'risk_adjusted_return': test_results['total_return'] / (test_results['max_drawdown'] + 1e-10)
                    }
                    
                    # Create result object
                    result = OptimizationResult(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        parameters=best_params,
                        metrics=test_results,
                        risk_metrics=risk_metrics
                    )
                    
                    results.append(result)
                    
                    print(f"\nWindow {i+1} Results:")
                    print(f"Train Period: {train_data.index[0]} to {train_data.index[-1]}")
                    print(f"Test Period: {test_data.index[0]} to {test_data.index[-1]}")
                    print(f"Best Parameters: {best_params}")
                    print(f"Test Performance:")
                    print(f"Total Return: {test_results['total_return']:.2f}%")
                    print(f"Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
                    print(f"Max Drawdown: {test_results['max_drawdown']:.2f}%")
                    print(f"Win Rate: {test_results['win_rate']:.2f}%")
                    
                except Exception as e:
                    print(f"Error in testing: {e}")
                    continue
                    
        return results
        
    def cross_validate_strategy(self,
                              strategy_name: str,
                              symbol: str,
                              timeframe: str,
                              parameters: Dict,
                              folds: int = 5,
                              days_per_fold: int = 30,
                              btc_price: float = None) -> Dict:
        """
        Perform k-fold cross validation of strategy parameters.
        
        Args:
            strategy_name: Name of strategy to test
            symbol: Trading pair symbol
            timeframe: Trading timeframe
            parameters: Strategy parameters to validate
            folds: Number of folds for cross validation
            days_per_fold: Number of days per fold
            btc_price: Current BTC price for volume calculations
            
        Returns:
            Dictionary containing cross validation metrics
        """
        # Get BTC price if not provided
        if btc_price is None:
            btc_price = self.data_fetcher.get_current_price('BTCUSDT')
            
        # Get historical data
        total_days = days_per_fold * folds
        end_time = datetime.now()
        start_time = end_time - timedelta(days=total_days)
        
        data = self.data_fetcher.get_historical_klines(
            symbol=symbol,
            interval=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Initialize metrics lists
        returns = []
        sharpe_ratios = []
        sortino_ratios = []
        max_drawdowns = []
        win_rates = []
        profit_factors = []
        calmar_ratios = []
        
        # Perform k-fold cross validation
        fold_size = len(data) // folds
        
        for i in range(folds):
            print(f"\nProcessing fold {i+1}/{folds}")
            
            # Get test fold
            test_start = i * fold_size
            test_end = test_start + fold_size
            test_data = data.iloc[test_start:test_end]
            
            try:
                # Initialize strategy and backtester
                strategy = self.strategies[strategy_name](params=parameters)
                backtester = Backtester(initial_capital=10000, commission=0.001)
                
                # Run backtest on fold
                results = backtester.run_backtest(strategy, test_data, btc_price)
                
                # Collect metrics
                returns.append(results['total_return'])
                sharpe_ratios.append(results['sharpe_ratio'])
                sortino_ratios.append(results['sortino_ratio'])
                max_drawdowns.append(results['max_drawdown'])
                win_rates.append(results['win_rate'])
                profit_factors.append(results['profit_factor'])
                calmar_ratios.append(
                    results['total_return'] / results['max_drawdown']
                    if results['max_drawdown'] > 0 else float('inf')
                )
                
                print(f"Fold {i+1} Results:")
                print(f"Total Return: {results['total_return']:.2f}%")
                print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
                print(f"Win Rate: {results['win_rate']:.2f}%")
                
            except Exception as e:
                print(f"Error in fold {i+1}: {e}")
                continue
                
        # Calculate cross validation metrics
        cv_metrics = {
            'total_return': {
                'mean': np.mean(returns) if returns else 0,
                'std': np.std(returns) if returns else 0,
                'min': np.min(returns) if returns else 0,
                'max': np.max(returns) if returns else 0
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'std': np.std(sharpe_ratios) if sharpe_ratios else 0,
                'min': np.min(sharpe_ratios) if sharpe_ratios else 0,
                'max': np.max(sharpe_ratios) if sharpe_ratios else 0
            },
            'sortino_ratio': {
                'mean': np.mean(sortino_ratios) if sortino_ratios else 0,
                'std': np.std(sortino_ratios) if sortino_ratios else 0,
                'min': np.min(sortino_ratios) if sortino_ratios else 0,
                'max': np.max(sortino_ratios) if sortino_ratios else 0
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns) if max_drawdowns else 0,
                'std': np.std(max_drawdowns) if max_drawdowns else 0,
                'min': np.min(max_drawdowns) if max_drawdowns else 0,
                'max': np.max(max_drawdowns) if max_drawdowns else 0
            },
            'win_rate': {
                'mean': np.mean(win_rates) if win_rates else 0,
                'std': np.std(win_rates) if win_rates else 0,
                'min': np.min(win_rates) if win_rates else 0,
                'max': np.max(win_rates) if win_rates else 0
            },
            'profit_factor': {
                'mean': np.mean(profit_factors) if profit_factors else 0,
                'std': np.std(profit_factors) if profit_factors else 0,
                'min': np.min(profit_factors) if profit_factors else 0,
                'max': np.max(profit_factors) if profit_factors else 0
            },
            'calmar_ratio': {
                'mean': np.mean(calmar_ratios) if calmar_ratios else 0,
                'std': np.std(calmar_ratios) if calmar_ratios else 0,
                'min': np.min(calmar_ratios) if calmar_ratios else 0,
                'max': np.max(calmar_ratios) if calmar_ratios else 0
            }
        }
        
        print("\nCross Validation Summary:")
        print(f"Total Return: {cv_metrics['total_return']['mean']:.2f}% ± {cv_metrics['total_return']['std']:.2f}%")
        print(f"Sharpe Ratio: {cv_metrics['sharpe_ratio']['mean']:.2f} ± {cv_metrics['sharpe_ratio']['std']:.2f}")
        print(f"Max Drawdown: {cv_metrics['max_drawdown']['mean']:.2f}% ± {cv_metrics['max_drawdown']['std']:.2f}%")
        print(f"Win Rate: {cv_metrics['win_rate']['mean']:.2f}% ± {cv_metrics['win_rate']['std']:.2f}%")
        
        return cv_metrics
