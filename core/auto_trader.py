import os
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from binance.client import Client
from .strategy_development import StrategyDevelopment
from models.database import Strategy, Trade, db
import json
import time
from threading import Thread, Event
from flask import current_app
from trading_executor import TradingExecutor

logger = logging.getLogger(__name__)

class AutoTrader:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize AutoTrader"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.strategy_dev = StrategyDevelopment(api_key, api_secret)
        self.executor = TradingExecutor(api_key, api_secret, mode='paper')
        self.stop_event = Event()
        self.strategy_thread = None
        self.monitor_thread = None
        self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        self.timeframes = ['1h', '4h', '1d']
        self.strategy_types = ['Momentum', 'Mean Reversion', 'Breakout']
        
        # Risk management settings
        self.max_risk_per_trade = 0.02  # Maximum 2% risk per trade
        self.max_risk_per_strategy = 0.05  # Maximum 5% risk across all trades for a strategy
        self.max_total_risk = 0.15  # Maximum 15% total portfolio risk
        self.volatility_scaling = True  # Scale position size based on volatility
        self.performance_based_sizing = True  # Adjust size based on strategy performance
        
        self.performance_criteria = {
            'sharpe_ratio': 1.5,  # Minimum Sharpe ratio
            'total_returns': 0.1,  # 10% minimum return
            'max_drawdown': -0.15  # 15% maximum drawdown
        }
        self.is_running = False
        
    def start(self):
        """Start the AutoTrader"""
        if not self.is_running:
            self.stop_event.clear()
            self.is_running = True
            
            # Get Flask app instance
            from flask import current_app
            self.app = current_app._get_current_object()
            
            # Start strategy development thread
            self.strategy_thread = Thread(target=self._develop_strategies_loop)
            self.strategy_thread.daemon = True
            self.strategy_thread.start()
            
            # Start monitoring thread
            self.monitor_thread = Thread(target=self._monitor_strategies_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("AutoTrader started")

    def stop(self):
        """Stop the AutoTrader"""
        logger.info("Stopping AutoTrader...")
        self.stop_event.set()
        
    def _develop_strategies_loop(self):
        """Continuously develop and test new strategies"""
        while not self.stop_event.is_set():
            try:
                with self.app.app_context():
                    for pair in self.trading_pairs:
                        for timeframe in self.timeframes:
                            for strategy_type in self.strategy_types:
                                # Generate strategy name
                                name = f"Auto_{strategy_type}_{pair}_{timeframe}_{int(time.time())}"
                                
                                # Develop and test strategy
                                try:
                                    logger.info(f"Developing strategy: {name}")
                                    strategy_result = self.strategy_dev.develop_strategy(
                                        name=name,
                                        strategy_type=strategy_type,
                                        symbol=pair,
                                        interval=timeframe,
                                        lookback_days=30
                                    )
                                    
                                    # Check if strategy meets performance criteria
                                    if self._evaluate_strategy(strategy_result['metrics']):
                                        logger.info(f"Strategy {name} meets performance criteria. Starting paper trading.")
                                        self._start_paper_trading(strategy_result['strategy_id'])
                                    else:
                                        logger.info(f"Strategy {name} does not meet performance criteria.")
                                        
                                except Exception as e:
                                    logger.error(f"Error developing strategy {name}: {str(e)}")
                                    
            except Exception as e:
                logger.error(f"Error in strategy development loop: {str(e)}")
                
            # Wait for 1 hour before next development cycle
            self.stop_event.wait(3600)
            
    def _monitor_strategies_loop(self):
        """Monitor and adjust running strategies"""
        while not self.stop_event.is_set():
            try:
                with self.app.app_context():
                    # Get all active strategies
                    active_strategies = Strategy.query.filter_by(is_active=True).all()
                    
                    for strategy in active_strategies:
                        try:
                            # Generate new signals
                            signals = self.strategy_dev.generate_signals(
                                strategy.symbol,
                                strategy.timeframe,
                                strategy.strategy_type
                            )
                            
                            if signals is not None and len(signals) > 0:
                                # Get the latest signal
                                latest_signal = signals.iloc[-1]
                                signal = latest_signal.get('position', 0)
                                
                                # Execute the signal if not 0
                                if signal != 0:
                                    risk_amount = self._calculate_risk_amount(strategy, strategy.symbol)
                                    if risk_amount > 0:
                                        trade_result = self.executor.execute_signal(
                                            strategy_id=strategy.id,
                                            symbol=strategy.symbol,
                                            signal=signal,
                                            risk_amount=risk_amount
                                        )
                                        
                                        if trade_result:
                                            logger.info(f"Trade executed for strategy {strategy.id}: {trade_result}")
                                    else:
                                        logger.warning(f"Skip trade for strategy {strategy.id} due to risk limits")
                            
                            # Calculate recent performance
                            recent_trades = Trade.query.filter_by(strategy_id=strategy.id)\
                                .filter(Trade.timestamp >= datetime.now() - timedelta(days=7))\
                                .all()
                            
                            if recent_trades:
                                performance = self._calculate_performance(recent_trades)
                                
                                # Stop strategy if performance is poor
                                if not self._evaluate_strategy(performance):
                                    logger.info(f"Stopping strategy {strategy.name} due to poor performance")
                                    strategy.is_active = False
                                    db.session.commit()
                                    
                        except Exception as e:
                            logger.error(f"Error monitoring strategy {strategy.id}: {str(e)}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error in strategy monitoring loop: {str(e)}")
                
            # Check every hour
            self.stop_event.wait(3600)
            
    def _evaluate_strategy(self, metrics: Dict) -> bool:
        """Evaluate if strategy meets performance criteria"""
        return (
            metrics['sharpe_ratio'] >= self.performance_criteria['sharpe_ratio'] and
            metrics['total_returns'] >= self.performance_criteria['total_returns'] and
            metrics['max_drawdown'] >= self.performance_criteria['max_drawdown']
        )
        
    def _calculate_performance(self, trades: List[Trade]) -> Dict:
        """Calculate performance metrics from trades"""
        returns = []
        for trade in trades:
            returns.append((trade.exit_price - trade.entry_price) / trade.entry_price)
            
        returns = np.array(returns)
        
        return {
            'total_returns': float(np.sum(returns)),
            'sharpe_ratio': float(np.sqrt(252) * np.mean(returns) / np.std(returns)) if len(returns) > 1 else 0,
            'max_drawdown': float(np.min(np.minimum.accumulate(returns))),
            'win_rate': float(np.mean(returns > 0))
        }
        
    def _calculate_risk_amount(self, strategy, symbol: str) -> float:
        """Calculate risk amount based on multiple factors"""
        try:
            # Get current account balance
            account_balance = self.executor.get_account_balance()
            
            # Base risk amount (2% of account)
            base_risk = account_balance * self.max_risk_per_trade
            
            # Adjust for strategy performance if enabled
            if self.performance_based_sizing:
                recent_trades = Trade.query.filter_by(strategy_id=strategy.id)\
                    .filter(Trade.timestamp >= datetime.now() - timedelta(days=30))\
                    .all()
                
                if recent_trades:
                    win_rate = sum(1 for t in recent_trades if t.pnl > 0) / len(recent_trades)
                    avg_profit = sum(t.pnl for t in recent_trades) / len(recent_trades)
                    
                    # Scale risk based on performance (0.5x to 1.5x)
                    performance_scalar = min(1.5, max(0.5, (win_rate * 2 + avg_profit/base_risk)))
                    base_risk *= performance_scalar
            
            # Adjust for volatility if enabled
            if self.volatility_scaling:
                # Calculate 14-day ATR
                klines = self.executor.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1DAY,
                    limit=14
                )
                prices = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close'])
                prices = prices.astype(float)
                
                # Calculate ATR
                tr1 = prices['high'] - prices['low']
                tr2 = abs(prices['high'] - prices['close'].shift())
                tr3 = abs(prices['low'] - prices['close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                
                # Normalize ATR to percentage
                current_price = float(prices['close'].iloc[-1])
                atr_pct = atr / current_price
                
                # Adjust risk based on volatility (0.5x to 1.5x)
                vol_scalar = min(1.5, max(0.5, 0.02 / atr_pct))  # 2% as baseline volatility
                base_risk *= vol_scalar
            
            # Check strategy risk limits
            strategy_positions = Trade.query.filter_by(
                strategy_id=strategy.id,
                status='open'
            ).all()
            current_strategy_risk = sum(t.risk_amount for t in strategy_positions)
            
            if current_strategy_risk + base_risk > account_balance * self.max_risk_per_strategy:
                base_risk = max(0, account_balance * self.max_risk_per_strategy - current_strategy_risk)
            
            # Check total portfolio risk limits
            all_positions = Trade.query.filter_by(status='open').all()
            total_risk = sum(t.risk_amount for t in all_positions)
            
            if total_risk + base_risk > account_balance * self.max_total_risk:
                base_risk = max(0, account_balance * self.max_total_risk - total_risk)
            
            return round(base_risk, 2)
            
        except Exception as e:
            logger.error(f"Error calculating risk amount: {str(e)}")
            return 0.0
        
    def _start_paper_trading(self, strategy_id: int):
        """Start paper trading for a strategy"""
        try:
            strategy = Strategy.query.get(strategy_id)
            if not strategy:
                logger.error(f"Strategy {strategy_id} not found")
                return
                
            strategy.is_active = True
            db.session.commit()
            
            # Generate initial signal
            signals = self.strategy_dev.generate_signals(
                strategy.symbol,
                strategy.timeframe,
                strategy.strategy_type
            )
            
            if signals is not None and len(signals) > 0:
                # Get the latest signal
                latest_signal = signals.iloc[-1]
                signal = latest_signal.get('position', 0)
                
                # Execute the signal
                if signal != 0:
                    risk_amount = self._calculate_risk_amount(strategy, strategy.symbol)
                    if risk_amount > 0:
                        trade_result = self.executor.execute_signal(
                            strategy_id=strategy_id,
                            symbol=strategy.symbol,
                            signal=signal,
                            risk_amount=risk_amount
                        )
                        
                        if trade_result:
                            logger.info(f"Initial trade executed for strategy {strategy_id}: {trade_result}")
                    else:
                        logger.warning(f"Skip trade for strategy {strategy_id} due to risk limits")
                    
        except Exception as e:
            logger.error(f"Error starting paper trading for strategy {strategy_id}: {str(e)}")

    def develop_strategy(self, strategy_id: int):
        """Develop a new trading strategy"""
        try:
            with self.app.app_context():
                strategy = Strategy.query.get(strategy_id)
                if not strategy:
                    logger.error(f"Strategy {strategy_id} not found")
                    return
                    
                # Generate initial signals
                signals = self.strategy_dev.generate_signals(
                    strategy.symbol,
                    strategy.timeframe,
                    strategy.strategy_type
                )
                
                if signals is not None and len(signals) > 0:
                    # Calculate performance metrics
                    total_returns = (signals['close'].iloc[-1] / signals['close'].iloc[0]) - 1
                    daily_returns = signals['close'].pct_change().dropna()
                    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
                    
                    # Calculate drawdown
                    cumulative_returns = (1 + daily_returns).cumprod()
                    rolling_max = cumulative_returns.expanding().max()
                    drawdowns = cumulative_returns / rolling_max - 1
                    max_drawdown = drawdowns.min()
                    
                    # Update strategy with performance metrics
                    strategy.backtesting_results = {
                        'total_return': float(total_returns * 100),
                        'sharpe_ratio': float(sharpe_ratio),
                        'max_drawdown': float(max_drawdown * 100)
                    }
                    
                    # Check if strategy meets performance criteria
                    if (total_returns >= 0.10 and  # 10% minimum return
                        sharpe_ratio >= 1.5 and    # Minimum Sharpe ratio
                        max_drawdown >= -0.15):    # Maximum 15% drawdown
                        
                        strategy.development_status = 'ready'
                        logger.info(f"Strategy {strategy.name} developed successfully")
                    else:
                        strategy.development_status = 'failed'
                        logger.info(f"Strategy {strategy.name} failed to meet performance criteria")
                        
                    db.session.commit()
                    
        except Exception as e:
            logger.error(f"Error developing strategy {strategy_id}: {str(e)}")
