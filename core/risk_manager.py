import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

@dataclass
class PositionSize:
    units: float
    value: float
    leverage: float = 1.0
    risk_amount: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

@dataclass
class RiskMetrics:
    volatility: float
    market_regime: str
    correlation_impact: float
    liquidity_score: float
    vol_z_score: float
    risk_score: float = 0.0  # Default risk score

    def calculate_risk_score(self):
        """Calculate a composite risk score based on metrics."""
        self.risk_score = (
            self.volatility +
            abs(self.correlation_impact) +
            (1 - self.liquidity_score) +
            abs(self.vol_z_score)
        ) / 4

class RiskManager:
    """Advanced risk management system."""
    
    def __init__(self,
                 max_position_size: float = 0.1,  # Max position size as fraction of capital
                 max_drawdown: float = 0.2,       # Maximum allowed drawdown
                 risk_per_trade: float = 0.02,    # Risk per trade as fraction of capital
                 vol_lookback: int = 20,          # Lookback period for volatility
                 vol_z_score: float = 2.0,        # Z-score for volatility scaling
                 correlation_threshold: float = 0.7):  # Correlation threshold for risk scaling
        
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        self.vol_lookback = vol_lookback
        self.vol_z_score = vol_z_score
        self.correlation_threshold = correlation_threshold
        
        self.positions = {}  # Track open positions
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()
        self.last_risk_metrics = None
        
    def calculate_risk_metrics(self, data: pd.DataFrame, current_price: float) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        returns = data['close'].pct_change().dropna()
        
        # Volatility metrics
        current_vol = returns.rolling(window=self.vol_lookback).std().iloc[-1]
        vol_z_score = (current_vol - returns.std()) / returns.std()
        
        # Market regime metrics
        regime = self._detect_market_regime(data)
        
        # Liquidity metrics
        volume_ma = data['volume'].rolling(window=self.vol_lookback).mean().iloc[-1]
        volume_std = data['volume'].rolling(window=self.vol_lookback).std().iloc[-1]
        liquidity_score = (data['volume'].iloc[-1] - volume_ma) / volume_std
        
        # Correlation metrics
        correlation = self._calculate_correlation_impact(data)
        
        risk_metrics = RiskMetrics(
            volatility=current_vol,
            market_regime=regime,
            correlation_impact=correlation,
            liquidity_score=liquidity_score,
            vol_z_score=vol_z_score
        )
        risk_metrics.calculate_risk_score()
        self.last_risk_metrics = risk_metrics
        return self.last_risk_metrics
        
    def calculate_position_size(self, capital: float, price: float, risk_metrics: RiskMetrics) -> float:
        """Calculate dynamic position size based on risk metrics."""
        # Base position size
        base_size = capital * self.risk_per_trade / price
        
        # Volatility adjustment
        vol_scalar = self._calculate_volatility_scalar(risk_metrics.vol_z_score)
        
        # Market regime adjustment
        regime_scalar = self._calculate_regime_scalar(risk_metrics.market_regime)
        
        # Liquidity adjustment
        liquidity_scalar = self._calculate_liquidity_scalar(risk_metrics.liquidity_score)
        
        # Correlation adjustment
        correlation_scalar = self._calculate_correlation_scalar(risk_metrics.correlation_impact)
        
        # Final position size
        position_size = base_size * vol_scalar * regime_scalar * liquidity_scalar * correlation_scalar
        
        # Apply maximum position size constraint
        max_size = capital * self.max_position_size / price
        return min(position_size, max_size)
        
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate dynamic stop loss level."""
        # Base stop loss percentage
        stop_loss_pct = self.risk_per_trade
        
        # Adjust for volatility
        if self.last_risk_metrics:
            vol_adjustment = self.last_risk_metrics.volatility * self.vol_z_score
            stop_loss_pct *= (1 + vol_adjustment)
        
        # Calculate stop loss price
        if is_long:
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)
            
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime."""
        returns = data['close'].pct_change().dropna()
        
        # Calculate trend
        sma_short = data['close'].rolling(window=20).mean()
        sma_long = data['close'].rolling(window=50).mean()
        trend = 1 if sma_short.iloc[-1] > sma_long.iloc[-1] else -1
        
        # Calculate volatility regime
        vol = returns.rolling(window=20).std()
        vol_percentile = vol.rank(pct=True).iloc[-1]
        
        if vol_percentile > 0.8:
            regime = 'high_volatility'
        elif vol_percentile < 0.2:
            regime = 'low_volatility'
        else:
            regime = 'normal_volatility'
            
        # Combine with trend
        if trend > 0:
            regime = f'uptrend_{regime}'
        else:
            regime = f'downtrend_{regime}'
            
        return regime
        
    def _calculate_correlation_impact(self, data: pd.DataFrame) -> float:
        """Calculate correlation impact on risk."""
        if len(data) < self.vol_lookback:
            return 1.0
            
        # Calculate correlation with market
        market_corr = data['close'].pct_change().corr(data['volume'].pct_change())
        
        # Scale correlation impact
        if abs(market_corr) > self.correlation_threshold:
            return 0.5  # Reduce position size when highly correlated
        return 1.0
        
    def _calculate_volatility_scalar(self, vol_z_score: float) -> float:
        """Calculate volatility-based position scalar."""
        if vol_z_score > self.vol_z_score:
            return 0.5  # Reduce position size in high volatility
        elif vol_z_score < -self.vol_z_score:
            return 1.5  # Increase position size in low volatility
        return 1.0
        
    def _calculate_regime_scalar(self, regime: str) -> float:
        """Calculate market regime-based position scalar."""
        regime_scalars = {
            'uptrend_normal_volatility': 1.0,
            'uptrend_low_volatility': 1.2,
            'uptrend_high_volatility': 0.8,
            'downtrend_normal_volatility': 0.8,
            'downtrend_low_volatility': 1.0,
            'downtrend_high_volatility': 0.6
        }
        return regime_scalars.get(regime, 0.8)
        
    def _calculate_liquidity_scalar(self, liquidity_score: float) -> float:
        """Calculate liquidity-based position scalar."""
        if liquidity_score < -2:
            return 0.5  # Reduce position size in low liquidity
        elif liquidity_score > 2:
            return 1.2  # Increase position size in high liquidity
        return 1.0
        
    def _calculate_correlation_scalar(self, correlation_impact: float) -> float:
        """Calculate correlation-based position scalar."""
        return correlation_impact  # Already scaled in _calculate_correlation_impact
        
    def update_position(self, position_id: str, current_price: float, timestamp: datetime):
        """Update position status and trailing stops."""
        if position_id not in self.positions:
            return
            
        position = self.positions[position_id]
        
        # Update trailing stop if price has moved in our favor
        if position.direction == 'long' and current_price > position.entry_price:
            new_stop = current_price * (1 - self.risk_per_trade)
            position.stop_loss = max(position.stop_loss, new_stop)
        elif position.direction == 'short' and current_price < position.entry_price:
            new_stop = current_price * (1 + self.risk_per_trade)
            position.stop_loss = min(position.stop_loss, new_stop)
            
    def reset_daily_metrics(self):
        """Reset daily risk metrics."""
        current_time = datetime.now()
        if current_time - self.last_reset > timedelta(days=1):
            self.daily_pnl = 0.0
            self.last_reset = current_time
