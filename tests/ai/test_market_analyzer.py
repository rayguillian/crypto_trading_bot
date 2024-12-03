import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai.market_analyzer import MarketAnalyzer

@pytest.fixture
def market_analyzer():
    """Create MarketAnalyzer instance."""
    return MarketAnalyzer()

@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 60000, 100),
        'high': np.random.uniform(41000, 61000, 100),
        'low': np.random.uniform(39000, 59000, 100),
        'close': np.random.uniform(40000, 60000, 100),
        'volume': np.random.uniform(100, 1000, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'macd': np.random.uniform(-100, 100, 100),
        'volatility': np.random.uniform(0.01, 0.05, 100)
    })

def test_market_analyzer_initialization(market_analyzer):
    """Test MarketAnalyzer initialization."""
    assert hasattr(market_analyzer, 'analyze_market_conditions')
    assert hasattr(market_analyzer, 'generate_trading_signals')
    assert hasattr(market_analyzer, 'get_market_sentiment')

def test_market_condition_analysis(market_analyzer, sample_market_data):
    """Test market condition analysis."""
    analysis = market_analyzer.analyze_market_conditions(sample_market_data)
    
    required_metrics = [
        'trend_strength',
        'volatility_regime',
        'market_regime',
        'support_resistance_levels',
        'key_price_levels'
    ]
    
    assert all(metric in analysis for metric in required_metrics)
    assert isinstance(analysis['trend_strength'], float)
    assert isinstance(analysis['market_regime'], str)

def test_trading_signal_generation(market_analyzer, sample_market_data):
    """Test trading signal generation."""
    signals = market_analyzer.generate_trading_signals(sample_market_data)
    
    assert 'entry_signals' in signals
    assert 'exit_signals' in signals
    assert 'confidence_scores' in signals
    
    # Check signal properties
    assert all(signal in [-1, 0, 1] for signal in signals['entry_signals'])
    assert all(0 <= score <= 1 for score in signals['confidence_scores'])

def test_market_sentiment_analysis(market_analyzer):
    """Test market sentiment analysis."""
    sentiment = market_analyzer.get_market_sentiment('BTCUSDT')
    
    assert 'sentiment_score' in sentiment
    assert 'sentiment_sources' in sentiment
    assert -1 <= sentiment['sentiment_score'] <= 1

def test_pattern_recognition(market_analyzer, sample_market_data):
    """Test technical pattern recognition."""
    patterns = market_analyzer.identify_patterns(sample_market_data)
    
    assert isinstance(patterns, list)
    for pattern in patterns:
        assert 'pattern_type' in pattern
        assert 'confidence' in pattern
        assert 'price_target' in pattern

def test_support_resistance_detection(market_analyzer, sample_market_data):
    """Test support and resistance level detection."""
    levels = market_analyzer.detect_support_resistance(sample_market_data)
    
    assert 'support_levels' in levels
    assert 'resistance_levels' in levels
    assert isinstance(levels['support_levels'], list)
    assert isinstance(levels['resistance_levels'], list)

def test_trend_analysis(market_analyzer, sample_market_data):
    """Test trend analysis capabilities."""
    trend_analysis = market_analyzer.analyze_trend(sample_market_data)
    
    required_metrics = [
        'trend_direction',
        'trend_strength',
        'trend_duration',
        'key_levels'
    ]
    
    assert all(metric in trend_analysis for metric in required_metrics)
    assert trend_analysis['trend_direction'] in ['bullish', 'bearish', 'sideways']

def test_volatility_analysis(market_analyzer, sample_market_data):
    """Test volatility analysis."""
    volatility_metrics = market_analyzer.analyze_volatility(sample_market_data)
    
    assert 'current_volatility' in volatility_metrics
    assert 'volatility_regime' in volatility_metrics
    assert 'volatility_forecast' in volatility_metrics
    assert isinstance(volatility_metrics['current_volatility'], float)

def test_market_regime_classification(market_analyzer, sample_market_data):
    """Test market regime classification."""
    regime = market_analyzer.classify_market_regime(sample_market_data)
    
    assert 'current_regime' in regime
    assert 'regime_probability' in regime
    assert 'regime_duration' in regime
    assert regime['current_regime'] in [
        'trending', 'ranging', 'volatile', 'breakout'
    ]

def test_ai_prediction_generation(market_analyzer, sample_market_data):
    """Test AI-based prediction generation."""
    predictions = market_analyzer.generate_predictions(sample_market_data)
    
    assert 'price_prediction' in predictions
    assert 'confidence_interval' in predictions
    assert 'prediction_horizon' in predictions
    assert isinstance(predictions['price_prediction'], float)

def test_correlation_analysis(market_analyzer):
    """Test correlation analysis between assets."""
    correlations = market_analyzer.analyze_correlations(['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
    
    assert isinstance(correlations, pd.DataFrame)
    assert correlations.shape[0] == correlations.shape[1]
    assert all(-1 <= corr <= 1 for corr in correlations.values.flatten())

def test_risk_reward_analysis(market_analyzer, sample_market_data):
    """Test risk/reward analysis for potential trades."""
    analysis = market_analyzer.analyze_risk_reward(
        entry_price=50000,
        target_price=55000,
        stop_loss=48000,
        market_data=sample_market_data
    )
    
    assert 'risk_reward_ratio' in analysis
    assert 'probability_of_success' in analysis
    assert 'expected_value' in analysis
    assert analysis['risk_reward_ratio'] > 0

def test_market_impact_analysis(market_analyzer):
    """Test market impact analysis for large orders."""
    impact = market_analyzer.analyze_market_impact(
        symbol='BTCUSDT',
        order_size=1.0,
        side='buy'
    )
    
    assert 'price_impact' in impact
    assert 'slippage_estimate' in impact
    assert 'recommended_sizing' in impact
    assert impact['price_impact'] >= 0
