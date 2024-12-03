import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.backtester import Backtester
from strategies.rsi_macd_strategy import RSIMACDStrategy

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 60000, 1000),
        'high': np.random.uniform(41000, 61000, 1000),
        'low': np.random.uniform(39000, 59000, 1000),
        'close': np.random.uniform(40000, 60000, 1000),
        'volume': np.random.uniform(100, 1000, 1000)
    })

@pytest.fixture
def backtester():
    """Create Backtester instance."""
    return Backtester(initial_capital=10000.0, commission=0.001)

def test_backtester_initialization(backtester):
    """Test backtester initialization."""
    assert backtester.initial_capital == 10000.0
    assert backtester.equity == 10000.0
    assert len(backtester.trades) == 0
    assert backtester.current_position is None

def test_run_backtest(backtester, sample_data):
    """Test running a complete backtest."""
    strategy = RSIMACDStrategy()
    results = backtester.run_backtest(strategy, sample_data)
    
    # Check results structure
    assert isinstance(results, dict)
    assert 'trades' in results
    assert 'equity_curve' in results
    
    # Check trade log
    trades_df = results['trades']
    assert all(col in trades_df.columns for col in [
        'entry_time', 'exit_time', 'entry_price', 'exit_price',
        'position_size', 'pnl', 'return'
    ])
    
    # Check equity curve
    equity_curve = results['equity_curve']
    assert len(equity_curve) > 0
    assert equity_curve.index.is_monotonic_increasing

def test_position_sizing(backtester, sample_data):
    """Test position sizing calculations."""
    # Test with different risk levels
    backtester.risk_per_trade = 0.02  # 2% risk per trade
    position_size = backtester._calculate_position_size(
        capital=10000,
        entry_price=50000,
        stop_loss=49000
    )
    
    assert 0 < position_size <= 0.2  # Position size should be reasonable
    assert position_size * 50000 <= 10000  # Position value shouldn't exceed capital

def test_risk_management(backtester, sample_data):
    """Test risk management rules."""
    # Set risk limits
    backtester.max_position_size = 0.2
    backtester.max_drawdown = 0.1
    
    # Run backtest with risk limits
    strategy = RSIMACDStrategy()
    results = backtester.run_backtest(strategy, sample_data)
    
    # Check if risk limits were respected
    assert results['equity_curve'].min() >= 10000 * (1 - 0.1)
    assert all(trade['position_size'] <= 0.2 for trade in results['trades'].to_dict('records'))

def test_performance_metrics(backtester, sample_data):
    """Test calculation of performance metrics."""
    strategy = RSIMACDStrategy()
    results = backtester.run_backtest(strategy, sample_data)
    equity_curve = results['equity_curve']
    
    # Check metric bounds
    assert equity_curve.max() >= equity_curve.min()
    
    # Check metric calculations
    trades = results['trades']
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    
    assert total_trades == len(trades)
    assert winning_trades <= total_trades

def test_trade_execution(backtester, sample_data):
    """Test trade execution logic."""
    # Execute a single trade
    entry_price = 50000
    position_size = 0.1
    
    backtester._execute_trade(
        timestamp=datetime.now(),
        entry_price=entry_price,
        position_size=position_size,
        direction=1
    )
    
    assert len(backtester.trades) == 1
    assert backtester.equity < backtester.initial_capital
    
    # Close the trade
    exit_price = 51000
    backtester._close_trade(
        timestamp=datetime.now() + timedelta(hours=1),
        exit_price=exit_price,
        position_id=0
    )
    
    assert len(backtester.trades) == 1
    assert backtester.equity > backtester.initial_capital

def test_drawdown_calculation(backtester):
    """Test drawdown calculations."""
    equity_curve = pd.Series([
        10000, 11000, 10500, 10200, 10800,
        10300, 10100, 10600, 10900, 11200
    ])
    
    drawdown = backtester._calculate_drawdown(equity_curve)
    max_drawdown = drawdown.max()
    
    assert isinstance(max_drawdown, float)
    assert 0 <= max_drawdown <= 1
    assert len(drawdown) == len(equity_curve)

def test_optimization_constraints(backtester, sample_data):
    """Test strategy optimization constraints."""
    constraints = {
        'min_trades': 10,
        'min_win_rate': 0.5,
        'min_profit': 0.05,
        'max_drawdown': 0.2,
        'min_sharpe': 1.0
    }
    
    strategy = RSIMACDStrategy()
    results = backtester.run_backtest(strategy, sample_data)
    equity_curve = results['equity_curve']
    
    # Check if results meet constraints
    meets_constraints = (
        len(results['trades']) >= constraints['min_trades'] and
        equity_curve.max() >= equity_curve.min() * (1 + constraints['min_profit']) and
        equity_curve.min() >= equity_curve.max() * (1 - constraints['max_drawdown'])
    )
    
    assert isinstance(meets_constraints, bool)

def test_parameter_optimization(backtester, sample_data):
    """Test parameter optimization process."""
    param_ranges = {
        'rsi_period': range(10, 30, 2),
        'macd_fast': range(8, 16, 2),
        'macd_slow': range(20, 30, 2)
    }
    
    best_params = backtester.optimize_parameters(
        data=sample_data,
        param_ranges=param_ranges,
        metric='sharpe_ratio'
    )
    
    assert isinstance(best_params, dict)
    assert all(param in best_params for param in param_ranges.keys())
    assert all(best_params[param] in param_ranges[param] for param in param_ranges.keys())
