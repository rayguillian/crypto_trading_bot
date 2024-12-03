import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .risk_manager import RiskManager, RiskMetrics

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    position_size: float
    direction: str  # 'long' or 'short'
    exit_time: Optional[datetime] = None  # Default to None
    exit_price: Optional[float] = None  # Default to None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    status: str = 'open'  # 'open' or 'closed'
    slippage: float = 0.0
    commission: float = 0.0
    entry_liquidity_score: float = 1.0
    exit_liquidity_score: float = 1.0
    market_impact: float = 0.0

class MarketSimulator:
    """Simulates realistic market conditions including slippage and liquidity."""
    
    def __init__(self, 
                 base_spread: float = 0.0002,  # 0.02% base spread
                 volatility_impact: float = 2.0,
                 min_liquidity_btc: float = 1000.0):
        self.base_spread = base_spread
        self.volatility_impact = volatility_impact
        self.min_liquidity_btc = min_liquidity_btc
    
    def calculate_execution_price(self,
                                price: float,
                                size: float,
                                volume: float,
                                volatility: float,
                                direction: str) -> Tuple[float, float]:
        """Calculate realistic execution price including slippage and market impact."""
        # Calculate spread based on volatility
        spread = self.base_spread * (1 + volatility * self.volatility_impact)
        
        # Calculate market impact
        market_impact = (size / volume) * price * 0.1  # Assume 10% price impact for using entire volume
        
        # Calculate liquidity score
        liquidity_score = min(1.0, volume / (self.min_liquidity_btc * price))
        
        # Calculate total slippage
        base_slippage = spread / 2
        volatility_slippage = volatility * 0.1 * (1 - liquidity_score)
        size_slippage = market_impact * (1 - liquidity_score)
        
        total_slippage = base_slippage + volatility_slippage + size_slippage
        
        # Adjust price based on direction
        if direction == 'long':
            execution_price = price * (1 + total_slippage)
        else:
            execution_price = price * (1 - total_slippage)
            
        return execution_price, market_impact

class Backtester:
    """Enhanced backtesting engine with realistic market simulation."""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,  # 0.1% commission
                 risk_manager: Optional[RiskManager] = None):
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_manager = risk_manager or RiskManager()
        self.market_simulator = MarketSimulator()
        self.reset()
        
    def reset(self):
        """Reset backtester state."""
        self.equity = self.initial_capital
        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None
        self.equity_curve = []
        self.drawdown_curve = []
        self.exposure_curve = []
        self.daily_returns = []
        
    def run_backtest(self, 
                    strategy: Any,
                    data: pd.DataFrame,
                    btc_price: float) -> Dict[str, Any]:
        """
        Run enhanced backtest with realistic market simulation.
        
        Args:
            strategy: Strategy instance with generate_signals method
            data: DataFrame with OHLCV data
            btc_price: Current BTC price for volume calculations
            
        Returns:
            Dictionary containing detailed backtest results and performance metrics
        """
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                logging.error("Data provided is not a valid DataFrame or is empty.")
                raise ValueError("Invalid or empty data provided")
                
            if not btc_price or btc_price <= 0:
                logging.error("BTC price is invalid: %s", btc_price)
                raise ValueError("Invalid BTC price provided")
                
            logging.info("Resetting backtester state.")
            self.reset()
            
            # Calculate signals
            signals = strategy.generate_signals(data)
            if signals is None or signals.empty:
                logging.error("Strategy failed to generate signals.")
                raise ValueError("Strategy failed to generate signals")
                
            logging.info("Signals generated successfully.")
            
            # Calculate volatility for the entire period
            returns = data['close'].pct_change()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                if i < 20:  # Skip first few rows for indicators to warm up
                    continue
                    
                logging.debug("Processing row %d: %s", i, row.to_dict())
                
                # Update equity curve
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': self.equity
                })
                
                # Calculate current volatility and volume metrics
                current_vol = rolling_vol.iloc[i]
                current_volume = row['volume']
                
                logging.debug("Current volatility: %f, Current volume: %f", current_vol, current_volume)
                
                # Get risk metrics
                risk_metrics = self.risk_manager.calculate_risk_metrics(
                    data.iloc[max(0, i-100):i+1],
                    btc_price
                )
                
                signal = signals.loc[timestamp]
                
                # Process exit signals first
                if self.current_position and (signal['exit'] or self._check_stop_conditions(row, risk_metrics)):
                    logging.info("Exiting position at %s", timestamp)
                    self._close_position(timestamp, row, current_vol, current_volume, risk_metrics)
                    
                # Then process entry signals
                if not self.current_position and not self._check_risk_limits(risk_metrics):
                    if signal['entry'] > 0:  # Long signal
                        logging.info("Entering long position at %s", timestamp)
                        self._open_position(timestamp, row, 'long', current_vol, current_volume, risk_metrics)
                    elif signal['entry'] < 0:  # Short signal
                        logging.info("Entering short position at %s", timestamp)
                        self._open_position(timestamp, row, 'short', current_vol, current_volume, risk_metrics)
                
                # Update metrics
                self._update_metrics(timestamp, row)
            
            # Close any remaining position
            if self.current_position:
                last_row = data.iloc[-1]
                logging.info("Closing remaining position at the end of data.")
                self._close_position(data.index[-1], last_row, rolling_vol.iloc[-1], last_row['volume'], risk_metrics)
            
            # Calculate and return performance metrics
            logging.info("Backtest completed successfully. Returning performance metrics.")
            performance_metrics = self._calculate_performance_metrics()
            logging.debug("Performance metrics: %s", performance_metrics)
            return performance_metrics
            
        except Exception as e:
            logging.error(f"Backtest failed: {str(e)}")
            return {
                'error': str(e),
                'trades': [],
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
    def _open_position(self, 
                      timestamp: datetime, 
                      row: pd.Series,
                      direction: str,
                      volatility: float,
                      volume: float,
                      risk_metrics: RiskMetrics):
        """Open a new position with realistic execution simulation."""
        # Calculate position size using risk manager
        position_size = self.risk_manager.calculate_position_size(
            self.equity,
            row['close'],
            risk_metrics
        )
        
        if not position_size:
            return
            
        # Calculate execution price with slippage
        execution_price, market_impact = self.market_simulator.calculate_execution_price(
            row['close'],
            position_size,
            volume,
            volatility,
            direction
        )
        
        # Create new trade
        self.current_position = Trade(
            entry_time=timestamp,
            entry_price=execution_price,
            position_size=position_size,
            direction=direction,
            market_impact=market_impact,
            slippage=(execution_price - row['close']) / row['close']
        )
        
        # Update equity
        commission = position_size * execution_price * self.commission
        self.equity -= commission
        
    def _close_position(self,
                       timestamp: datetime,
                       row: pd.Series,
                       volatility: float,
                       volume: float,
                       risk_metrics: RiskMetrics):
        """Close position with realistic execution simulation."""
        if not self.current_position:
            return
            
        # Calculate execution price with slippage
        execution_price, market_impact = self.market_simulator.calculate_execution_price(
            row['close'],
            self.current_position.position_size,
            volume,
            volatility,
            'short' if self.current_position.direction == 'long' else 'long'
        )
        
        # Update trade object
        self.current_position.exit_time = timestamp
        self.current_position.exit_price = execution_price
        self.current_position.status = 'closed'
        self.current_position.market_impact += market_impact
        
        # Calculate PnL
        if self.current_position.direction == 'long':
            pnl = (execution_price - self.current_position.entry_price) * self.current_position.position_size
        else:
            pnl = (self.current_position.entry_price - execution_price) * self.current_position.position_size
            
        pnl -= (self.current_position.commission + execution_price * self.current_position.position_size * self.commission)
        
        self.current_position.pnl = pnl
        self.current_position.pnl_percent = pnl / (self.current_position.entry_price * self.current_position.position_size)
        
        # Update equity and save trade
        self.equity += pnl - execution_price * self.current_position.position_size * self.commission
        self.trades.append(self.current_position)
        self.current_position = None
        
    def _check_stop_conditions(self, row: pd.Series, risk_metrics: RiskMetrics) -> bool:
        """Check for stop conditions including volatility-adjusted stops."""
        if not self.current_position:
            return False
            
        # Get current position details
        pos = self.current_position
        current_price = row['close']
        
        # Calculate dynamic stop distance based on volatility
        atr = self._calculate_atr(row)
        stop_distance = atr * self._get_stop_multiplier(risk_metrics)
        
        if pos.direction == 'long':
            stop_price = pos.entry_price - stop_distance
            return current_price <= stop_price
        else:
            stop_price = pos.entry_price + stop_distance
            return current_price >= stop_price
            
    def _check_risk_limits(self, risk_metrics: RiskMetrics) -> bool:
        """Check if risk limits would be exceeded."""
        return risk_metrics.risk_score > 0.8
        
    def _update_metrics(self, timestamp: datetime, row: pd.Series):
        """Update performance metrics."""
        # Calculate drawdown
        if self.equity_curve:
            peak = max(point['equity'] for point in self.equity_curve)
            drawdown = (peak - self.equity) / peak
            self.drawdown_curve.append({
                'timestamp': timestamp,
                'drawdown': drawdown
            })
        
        # Calculate exposure
        exposure = 0
        if self.current_position:
            exposure = self.current_position.position_size * row['close'] / self.equity
            
        self.exposure_curve.append({
            'timestamp': timestamp,
            'exposure': exposure
        })
        
        # Calculate daily returns
        if self.equity_curve and timestamp.date() != self.equity_curve[-1]['timestamp'].date():
            daily_return = (self.equity / self.equity_curve[-1]['equity']) - 1
            self.daily_returns.append(daily_return)
            
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'portfolio_value': [self.initial_capital],
                'trades': [],
                'total_return': 0.0
            }
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = pd.Series([t.pnl_percent for t in self.trades if t.pnl_percent is not None])
        
        # Calculate win rate
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        win_rate = winning_trades / len(self.trades) * 100
        
        # Calculate profit factor
        gross_profit = sum([t.pnl for t in self.trades if t.pnl > 0])
        gross_loss = abs(sum([t.pnl for t in self.trades if t.pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Calculate maximum drawdown
        portfolio_values = equity_df['equity'].values
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        logging.info("Calculating total return.")
        total_return = (portfolio_values[-1] / self.initial_capital - 1) * 100
        logging.debug("Total return calculated: %f", total_return)
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'portfolio_value': portfolio_values.tolist(),
            'trades': self.trades,
            'total_return': total_return
        }
