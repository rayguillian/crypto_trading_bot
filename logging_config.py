import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler with DEBUG level for detailed logs
    file_handler = RotatingFileHandler(
        'logs/trading_bot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Create strategy-specific logger
    strategy_logger = logging.getLogger('strategy')
    strategy_file_handler = RotatingFileHandler(
        'logs/strategy.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    strategy_file_handler.setFormatter(file_formatter)
    strategy_logger.addHandler(strategy_file_handler)
    strategy_logger.setLevel(logging.DEBUG)
