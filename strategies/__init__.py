from .base_strategy import BaseStrategy
from .rsi_macd_strategy import RSIMACDStrategy
from .bollinger_rsi_strategy import BollingerRSIStrategy
from .ma_crossover_strategy import MACrossoverStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .moving_average_strategy import MovingAverageStrategy
from .bbands_volume_strategy import BBandsVolumeStrategy
from .kama_strategy import KAMAStrategy
from .trend_volume_strategy import TrendVolumeStrategy
from .machine_learning_strategy import MachineLearningStrategy
from .advanced_strategies import AdvancedStrategy

__all__ = [
    'BaseStrategy',
    'RSIMACDStrategy',
    'BollingerRSIStrategy',
    'MACrossoverStrategy',
    'MeanReversionStrategy',
    'MovingAverageStrategy',
    'BBandsVolumeStrategy',
    'KAMAStrategy',
    'TrendVolumeStrategy',
    'MachineLearningStrategy',
    'AdvancedStrategy'
]

def get_strategy_class(strategy_name):
    strategies = {
        'RSIMACDStrategy': RSIMACDStrategy,
        'BollingerRSIStrategy': BollingerRSIStrategy,
        'MACrossoverStrategy': MACrossoverStrategy,
        'MeanReversionStrategy': MeanReversionStrategy,
        'MovingAverageStrategy': MovingAverageStrategy,
        'BBandsVolumeStrategy': BBandsVolumeStrategy,
        'KAMAStrategy': KAMAStrategy,
        'TrendVolumeStrategy': TrendVolumeStrategy,
        'MachineLearningStrategy': MachineLearningStrategy,
        'AdvancedStrategy': AdvancedStrategy
    }
    return strategies.get(strategy_name)
