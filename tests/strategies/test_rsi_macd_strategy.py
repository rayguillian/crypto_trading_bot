import pytest
import pandas as pd
import numpy as np
from strategies.rsi_macd_strategy import RSIMACDStrategy

@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H')
    np.random.seed(42)
    
    # Generate realistic price data
    close = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
    high = close * (1 + abs(np.random.randn(len(dates)) * 0.01))
    low = close * (1 - abs(np.random.randn(len(dates)) * 0.01))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close * (1 + np.random.randn(len(dates)) * 0.01),
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, size=len(dates))
    })
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def strategy():
    """Create RSI-MACD strategy instance."""
    return RSIMACDStrategy()

def test_strategy_initialization():
    """Test strategy initialization with default parameters."""
    strategy = RSIMACDStrategy()
    assert strategy.params['rsi_period'] == 21
    assert strategy.params['rsi_overbought'] == 65
    assert strategy.params['rsi_oversold'] == 35
    assert strategy.params['macd_fast'] == 10
    assert strategy.params['macd_slow'] == 20
    assert strategy.params['macd_signal'] == 9
    assert strategy.params['atr_period'] == 14
    assert strategy.params['atr_multiplier'] == 2.5

def test_indicator_calculation(sample_data):
    """Test calculation of technical indicators."""
    strategy = RSIMACDStrategy()
    df = strategy.calculate_indicators(sample_data)
    
    # Check if all required indicators are present
    assert 'rsi' in df.columns
    assert 'macd' in df.columns
    assert 'macd_signal' in df.columns
    assert 'macd_diff' in df.columns
    assert 'atr' in df.columns
    assert 'volatility' in df.columns
    
    # Verify indicator values are within expected ranges
    assert df['rsi'].between(0, 100).all()
    assert not df['atr'].isna().all()
    assert not df['volatility'].isna().all()

def test_signal_generation(sample_data):
    """Test trading signal generation."""
    strategy = RSIMACDStrategy()
    signals = strategy.generate_signals(sample_data)
    
    # Check signal DataFrame structure
    assert 'entry' in signals.columns
    assert 'exit' in signals.columns
    
    # Verify signal values
    assert signals['entry'].isin([0, 1, -1]).all()
    assert signals['exit'].dtype == bool

def test_get_signal(strategy, sample_data):
    """Test getting signal for current market conditions."""
    signal = strategy.get_signal(sample_data, len(sample_data) - 1)
    
    # Check signal attributes
    assert hasattr(signal, 'entry')
    assert hasattr(signal, 'exit')
    assert hasattr(signal, 'stop_loss')
    assert hasattr(signal, 'take_profit')
    assert hasattr(signal, 'position_size')
    
    # Check signal values
    assert signal.entry in [0, 1.0, -1.0]
    assert isinstance(signal.exit, bool)
    assert signal.stop_loss is None or isinstance(signal.stop_loss, float)
    assert signal.take_profit is None or isinstance(signal.take_profit, float)
    assert signal.position_size is None or isinstance(signal.position_size, float)

def test_parameter_validation(strategy):
    """Test parameter validation."""
    # Test valid parameters
    assert strategy.validate_parameters()
    
    # Test invalid RSI period
    strategy.params['rsi_period'] = 3
    assert not strategy.validate_parameters()
    
    # Test invalid MACD parameters
    strategy.params['rsi_period'] = 21
    strategy.params['macd_fast'] = 25
    strategy.params['macd_slow'] = 20
    assert not strategy.validate_parameters()
    
    # Test invalid ATR parameters
    strategy.params['macd_fast'] = 10
    strategy.params['atr_multiplier'] = 0.1
    assert not strategy.validate_parameters()

def test_stop_loss_calculation(strategy, sample_data):
    """Test stop loss calculation."""
    df = strategy.calculate_indicators(sample_data)
    stop_loss = strategy._calculate_stop_loss(df, len(df) - 1)
    
    assert isinstance(stop_loss, float)
    assert stop_loss < df['close'].iloc[-1]  # Stop loss should be below current price

def test_take_profit_calculation(strategy, sample_data):
    """Test take profit calculation."""
    df = strategy.calculate_indicators(sample_data)
    take_profit = strategy._calculate_take_profit(df, len(df) - 1)
    
    assert isinstance(take_profit, float)
    assert take_profit > df['close'].iloc[-1]  # Take profit should be above current price

def test_position_size_calculation(strategy, sample_data):
    """Test position size calculation."""
    df = strategy.calculate_indicators(sample_data)
    position_size = strategy._calculate_position_size(df, len(df) - 1)
    
    assert isinstance(position_size, float)
    assert 0 <= position_size <= 1.0  # Position size should be between 0 and 1
