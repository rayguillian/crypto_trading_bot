import pandas as pd
from core.strategy import BaseStrategy

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, name="MA Crossover", short_window=20, long_window=50):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """Generate trading signals based on MA crossover"""
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        return signals