import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from core.risk_manager import RiskManager, PositionSize

@pytest.fixture
def risk_manager():
    """Create RiskManager instance with default settings."""
    return RiskManager(
        max_position_size=0.02,
        max_daily_drawdown=0.05,
        max_total_positions=3
    )

@pytest.fixture
def sample_portfolio():
    """Create sample portfolio data."""
    return {
        'BTCUSDT': {'size': 0.1, 'entry_price': 50000},
        'ETHUSDT': {'size': 0.15, 'entry_price': 3000}
    }

@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1H'),
        'close': np.random.uniform(40000, 60000, 100),
        'volatility': np.random.uniform(0.01, 0.05, 100)
    })

def test_risk_manager_initialization(risk_manager):
    """Test RiskManager initialization."""
    assert risk_manager.max_position_size == 0.02
    assert risk_manager.max_daily_drawdown == 0.05
    assert risk_manager.max_total_positions == 3
    assert risk_manager.daily_pnl == 0.0
    assert risk_manager.open_positions == 0

def test_position_size_calculation(risk_manager):
    """Test position size calculation."""
    account_balance = 10000
    entry_price = 50000
    stop_loss = 49000
    exchange_info = {
        'min_qty': 0.001,
        'max_qty': 100,
        'step_size': 0.001
    }
    
    position = risk_manager.calculate_position_size(
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss=stop_loss,
        exchange_info=exchange_info
    )
    
    assert isinstance(position, PositionSize)
    assert position.units > 0
    assert position.value <= account_balance * risk_manager.max_position_size

def test_portfolio_risk_check(risk_manager, sample_portfolio):
    """Test portfolio risk assessment."""
    current_prices = {
        'BTCUSDT': 51000,
        'ETHUSDT': 2900
    }
    
    risk_metrics = risk_manager.check_portfolio_risk(
        portfolio=sample_portfolio,
        current_prices=current_prices
    )
    
    assert 'total_exposure' in risk_metrics
    assert 'current_drawdown' in risk_metrics
    assert 'risk_level' in risk_metrics
    assert isinstance(risk_metrics['risk_level'], str)

def test_stop_loss_calculation(risk_manager, sample_market_data):
    """Test dynamic stop loss calculation."""
    entry_price = 50000
    position_size = 0.1
    
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=entry_price,
        position_size=position_size,
        volatility=sample_market_data['volatility'].iloc[-1]
    )
    
    assert stop_loss < entry_price
    assert (entry_price - stop_loss) / entry_price <= 0.1  # Max 10% stop loss

def test_position_adjustment(risk_manager, sample_portfolio):
    """Test position size adjustment based on risk."""
    symbol = 'BTCUSDT'
    desired_position_size = 0.2
    current_price = 52000
    
    adjusted_size = risk_manager.adjust_position_size(
        symbol=symbol,
        desired_size=desired_position_size,
        current_portfolio=sample_portfolio,
        current_price=current_price
    )
    
    assert adjusted_size <= desired_position_size
    assert adjusted_size * current_price <= risk_manager.max_position_size * 10000

def test_drawdown_monitoring(risk_manager):
    """Test drawdown monitoring and alerts."""
    equity_curve = pd.Series([
        10000, 9900, 9800, 9700, 9600,  # 4% drawdown
        9700, 9800, 9850, 9900, 9950    # Recovery
    ])
    
    drawdown = risk_manager.monitor_drawdown(equity_curve)
    
    assert isinstance(drawdown, float)
    assert 0 <= drawdown <= 1
    assert drawdown <= risk_manager.max_daily_drawdown  # Should pass with 4% drawdown

def test_risk_exposure_limits(risk_manager, sample_portfolio):
    """Test risk exposure limits."""
    new_position = {
        'symbol': 'ADAUSDT',
        'size': 0.1,
        'entry_price': 1.5
    }
    
    # Check if new position would exceed limits
    can_add_position = risk_manager.check_position_limits(
        new_position=new_position,
        current_portfolio=sample_portfolio
    )
    
    assert isinstance(can_add_position, bool)
    if len(sample_portfolio) >= risk_manager.max_total_positions:
        assert not can_add_position

def test_volatility_adjustment(risk_manager, sample_market_data):
    """Test volatility-based position sizing."""
    base_position_size = 0.1
    
    adjusted_size = risk_manager.adjust_for_volatility(
        base_size=base_position_size,
        volatility=sample_market_data['volatility'].iloc[-1]
    )
    
    assert adjusted_size <= base_position_size
    assert adjusted_size > 0

def test_correlation_risk(risk_manager):
    """Test correlation-based risk assessment."""
    returns = pd.DataFrame({
        'BTCUSDT': np.random.normal(0, 0.02, 100),
        'ETHUSDT': np.random.normal(0, 0.03, 100),
        'ADAUSDT': np.random.normal(0, 0.04, 100)
    })
    
    correlation_risk = risk_manager.assess_correlation_risk(returns)
    
    assert isinstance(correlation_risk, float)
    assert -1 <= correlation_risk <= 1

def test_risk_report_generation(risk_manager, sample_portfolio, sample_market_data):
    """Test risk report generation."""
    report = risk_manager.generate_risk_report(
        portfolio=sample_portfolio,
        market_data=sample_market_data
    )
    
    required_metrics = [
        'total_exposure',
        'portfolio_beta',
        'value_at_risk',
        'current_drawdown',
        'position_sizes',
        'risk_concentration'
    ]
    
    assert all(metric in report for metric in required_metrics)
    assert isinstance(report['value_at_risk'], float)
    assert isinstance(report['risk_concentration'], dict)
