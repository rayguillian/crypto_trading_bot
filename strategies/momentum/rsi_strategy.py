import pandas as pd
from typing import Dict, Any
from ..base_strategy import BaseStrategy
from ta.momentum import RSIIndicator
import logging

class RSIStrategy(BaseStrategy):
    """RSI (Relative Strength Index) strategy."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'rsi_period': 14,
            'overbought': 70,
            'oversold': 30,
            'risk_per_trade': 1.0,
            'atr_period': 14,
            'atr_multiplier': 2.0
        }
        super().__init__(parameters or default_params)
    
    def prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        rsi = RSIIndicator(
            close=df['close'],
            window=self.parameters['rsi_period']
        )
        df['rsi'] = rsi.rsi()
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.prepare_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if pd.isna(df['rsi'].iloc[i]):
                continue
                
            rsi = df['rsi'].iloc[i]
            prev_rsi = df['rsi'].iloc[i-1]
            
            # Generate long signal
            if prev_rsi <= self.parameters['oversold'] and rsi > self.parameters['oversold']:
                signals.iloc[i] = 1
            # Generate short signal
            elif prev_rsi >= self.parameters['overbought'] and rsi < self.parameters['overbought']:
                signals.iloc[i] = -1
                
        return signals
