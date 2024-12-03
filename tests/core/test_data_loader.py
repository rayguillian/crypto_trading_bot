import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.data_loader import DataLoader

@pytest.fixture
def data_loader():
    """Create DataLoader instance."""
    return DataLoader()

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data."""
    now = datetime.now()
    dates = [now - timedelta(hours=x) for x in range(100)]
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 60000, 100),
        'high': np.random.uniform(41000, 61000, 100),
        'low': np.random.uniform(39000, 59000, 100),
        'close': np.random.uniform(40000, 60000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
    np.random.seed(42)
    
    # Generate base price to ensure valid OHLCV relationships
    base_price = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
    
    # Generate OHLCV data with valid price relationships
    df = pd.DataFrame({
        'timestamp': dates,
        'open': base_price,
        'close': base_price * (1 + np.random.randn(len(dates)) * 0.001),
        'volume': np.random.randint(1000, 10000, size=len(dates))
    })
    
    # Ensure high is highest and low is lowest
    df['high'] = np.maximum(
        df['open'],
        df['close']
    ) * (1 + abs(np.random.randn(len(dates)) * 0.001))
    
    df['low'] = np.minimum(
        df['open'],
        df['close']
    ) * (1 - abs(np.random.randn(len(dates)) * 0.001))
    
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def data_with_missing_values(sample_data):
    """Create data with missing values for testing."""
    df = sample_data.copy()
    mask = np.random.random(len(df)) < 0.1
    df.loc[mask, ['close', 'volume']] = np.nan
    return df

def test_data_loader_initialization(data_loader):
    """Test DataLoader initialization."""
    assert hasattr(data_loader, 'get_historical_data')
    assert hasattr(data_loader, 'get_live_data')
    assert hasattr(data_loader, '_calculate_atr')

def test_fetch_historical_data(data_loader):
    """Test historical data fetching."""
    symbol = 'BTCUSDT'
    interval = '4h'
    limit = 100
    
    data = data_loader.get_historical_data(symbol, interval, limit=limit)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) <= limit
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_calculate_indicators(data_loader, sample_ohlcv_data):
    """Test technical indicator calculation."""
    data = data_loader.get_historical_data('BTCUSDT', '4h', limit=100)
    
    # Check calculated indicators
    required_indicators = ['returns', 'volatility', 'atr', 'volume_ma']
    
    for indicator in required_indicators:
        assert indicator in data.columns
        assert not data[indicator].isnull().all()

def test_preprocess_data(data_loader, sample_data):
    """Test data preprocessing pipeline."""
    processed_data = data_loader.preprocess_data(sample_data)
    
    # Check that all required columns are present
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    assert all(col in processed_data.columns for col in required_cols)
    
    # Check that normalized columns are present
    norm_cols = ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm']
    assert all(col in processed_data.columns for col in norm_cols)
    
    # Verify no missing values
    assert not processed_data.isnull().any().any()

def test_validate_data(data_loader, sample_data):
    """Test data validation."""
    # Valid data should pass
    validated_data = data_loader.validate_data(sample_data)
    assert validated_data is not None
    
    # Test invalid price relationships
    invalid_data = sample_data.copy()
    invalid_data.loc[invalid_data.index[0], 'high'] = invalid_data.loc[invalid_data.index[0], 'low'] - 1
    with pytest.raises(ValueError, match="Invalid price relationships detected"):
        data_loader.validate_data(invalid_data)
    
    # Test missing columns
    missing_cols_data = sample_data.drop('volume', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        data_loader.validate_data(missing_cols_data)

def test_resample_data(data_loader, sample_data):
    """Test data resampling."""
    # Test upsampling
    resampled = data_loader.resample_data(sample_data, '4h')
    
    # Check that index frequency changed
    assert resampled.index.freq == pd.Timedelta('4h')
    
    # Verify OHLCV columns
    assert all(col in resampled.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Check technical indicators
    assert all(col in resampled.columns for col in ['returns', 'volatility', 'atr', 'volume_ma'])
    
    # Verify data integrity
    assert resampled['high'].max() <= sample_data['high'].max()
    assert resampled['low'].min() >= sample_data['low'].min()

def test_merge_data_sources(data_loader):
    """Test merging multiple data sources."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    
    # Create sample data from different sources
    source1 = pd.DataFrame({
        'open': np.random.uniform(40000, 60000, 100),
        'high': np.random.uniform(41000, 61000, 100),
        'low': np.random.uniform(39000, 59000, 100),
        'close': np.random.uniform(40000, 60000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    }, index=dates)
    
    source2 = pd.DataFrame({
        'sentiment': np.random.uniform(-1, 1, 100),
        'market_cap': np.random.uniform(1e9, 2e9, 100)
    }, index=dates)
    
    merged_data = data_loader.merge_data_sources([source1, source2])
    
    # Check that all columns are present
    assert all(col in merged_data.columns for col in source1.columns)
    assert all(col in merged_data.columns for col in source2.columns)
    
    # Check number of rows matches
    assert len(merged_data) == len(source1)
    
    # Test empty list
    with pytest.raises(ValueError, match="No DataFrames provided"):
        data_loader.merge_data_sources([])

def test_handle_missing_data(data_loader, data_with_missing_values):
    """Test handling of missing data."""
    handled_data = data_loader.handle_missing_data(data_with_missing_values)
    
    # Verify no missing values
    assert not handled_data.isnull().any().any()
    
    # Check that OHLCV data is forward filled
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in data_with_missing_values.columns:
            assert handled_data[col].equals(data_with_missing_values[col].fillna(method='ffill'))

def test_normalize_data(data_loader, sample_data):
    """Test data normalization."""
    normalized = data_loader.normalize_data(sample_data)
    
    # Check normalized columns exist
    norm_cols = ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm']
    assert all(col in normalized.columns for col in norm_cols)
    
    # Verify normalization ranges for price columns
    for col in ['open_norm', 'high_norm', 'low_norm', 'close_norm']:
        assert -1 <= normalized[col].min() <= normalized[col].max() <= 1
    
    # Verify volume normalization
    assert 0 <= normalized['volume_norm'].min() <= normalized['volume_norm'].max() <= 1

def test_preprocess_data(data_loader, sample_ohlcv_data):
    """Test data preprocessing."""
    processed_data = data_loader.preprocess_data(sample_ohlcv_data)
    
    # Check data cleaning
    assert not processed_data.isnull().any().any()
    assert processed_data.index.is_monotonic_increasing
    
    # Check calculated fields
    assert 'returns' in processed_data.columns
    assert 'volatility' in processed_data.columns
    
    # Check data types
    assert pd.api.types.is_float_dtype(processed_data['returns'])
    assert pd.api.types.is_float_dtype(processed_data['volatility'])

def test_data_validation(data_loader, sample_ohlcv_data):
    """Test data validation."""
    # Test with valid data
    assert data_loader.validate_data(sample_ohlcv_data)
    
    # Test with missing columns
    invalid_data = sample_ohlcv_data.drop('close', axis=1)
    with pytest.raises(ValueError):
        data_loader.validate_data(invalid_data)
    
    # Test with null values
    invalid_data = sample_ohlcv_data.copy()
    invalid_data.loc[0, 'close'] = None
    with pytest.raises(ValueError):
        data_loader.validate_data(invalid_data)

def test_resample_data(data_loader, sample_ohlcv_data):
    """Test data resampling."""
    # Test upsampling
    resampled_data = data_loader.resample_data(sample_ohlcv_data, '1h')
    assert len(resampled_data) >= len(sample_ohlcv_data)
    
    # Test downsampling
    resampled_data = data_loader.resample_data(sample_ohlcv_data, '12h')
    assert len(resampled_data) <= len(sample_ohlcv_data)
    
    # Check OHLCV integrity
    assert all(col in resampled_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_merge_data_sources(data_loader):
    """Test merging multiple data sources."""
    # Create sample data from different sources
    source1 = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'close': np.random.uniform(40000, 60000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    source2 = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'sentiment': np.random.uniform(-1, 1, 100),
        'market_cap': np.random.uniform(1e9, 2e9, 100)
    })
    
    merged_data = data_loader.merge_data_sources([source1, source2])
    
    assert len(merged_data) == len(source1)
    assert all(col in merged_data.columns for col in ['close', 'volume', 'sentiment', 'market_cap'])

def test_handle_missing_data(data_loader, sample_ohlcv_data):
    """Test handling of missing data."""
    # Create data with gaps
    data_with_gaps = sample_ohlcv_data.copy()
    data_with_gaps.loc[10:15, 'close'] = None
    
    filled_data = data_loader.handle_missing_data(data_with_gaps)
    
    assert not filled_data.isnull().any().any()
    assert len(filled_data) == len(data_with_gaps)

def test_data_normalization(data_loader, sample_ohlcv_data):
    """Test data normalization."""
    normalized_data = data_loader.normalize_data(sample_ohlcv_data)
    
    # Check if values are normalized
    for col in ['open', 'high', 'low', 'close']:
        assert -1 <= normalized_data[col].min() <= normalized_data[col].max() <= 1
