from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    def __init__(self, name):
        self.name = name
        self.positions = []
        self.trades = []
        
    @abstractmethod
    def generate_signals(self, data):
        """Generate trading signals based on data"""
        pass
        
    def backtest(self, data, initial_capital=10000.0):
        signals = self.generate_signals(data)
        equity_curve = []
        trades = []
        
        capital = initial_capital
        position = None
        
        for index, row in data.iterrows():
            signal = signals.loc[index]
            
            if signal == 1 and position is None:  # Buy signal
                position = {
                    'type': 'long',
                    'entry_date': index,
                    'entry_price': row['close'],
                    'size': capital / row['close']
                }
                
            elif signal == -1 and position:  # Sell signal
                pnl = (row['close'] - position['entry_price']) * position['size']
                capital += pnl
                
                trades.append({
                    'date': index,
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': row['close'],
                    'pnl': (pnl / initial_capital) * 100
                })
                
                position = None
            
            equity_curve.append({
                'date': index,
                'equity': capital if position is None else 
                         capital + (row['close'] - position['entry_price']) * position['size']
            })
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': self._calculate_metrics(trades, equity_curve, initial_capital)
        }
    
    def _calculate_metrics(self, trades, equity_curve, initial_capital):
        if not trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
            
        equity_series = pd.Series([e['equity'] for e in equity_curve])
        returns = equity_series.pct_change().dropna()
        
        win_trades = sum(1 for t in trades if t['pnl'] > 0)
        
        return {
            'total_return': (equity_series.iloc[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
            'max_drawdown': (equity_series / equity_series.cummax() - 1).min(),
            'win_rate': win_trades / len(trades) if trades else 0,
            'profit_factor': sum(t['pnl'] for t in trades if t['pnl'] > 0) / 
                            abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) 
                            if sum(t['pnl'] for t in trades if t['pnl'] < 0) != 0 else 0
        }