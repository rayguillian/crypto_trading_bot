from models.database import db, Strategy, Trade
from datetime import datetime, timedelta
import random
import numpy as np
from app import app

def init_test_strategies():
    with app.app_context():
        # Create database tables
        db.create_all()
        
        # Clear existing data
        Strategy.query.delete()
        Trade.query.delete()
        db.session.commit()
        
        # Test strategy configurations
        strategies = [
            {
                'name': 'MA Crossover Strategy',
                'description': 'Moving average crossover strategy using 20 and 50 period MAs',
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'strategy_type': 'trend_following',
                'development_status': 'live',
                'is_active': True,
                'is_starred': True,
                'backtesting_results': {
                    'total_trades': 150,
                    'winning_trades': 90,
                    'losing_trades': 60,
                    'total_return': 28.5,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': 12.5,
                    'win_rate': 0.60,
                    'profit_factor': 1.75,
                    'calmar_ratio': 2.28
                }
            },
            {
                'name': 'RSI Strategy',
                'description': 'Mean reversion strategy using RSI indicator',
                'symbol': 'ETHUSDT',
                'timeframe': '4h',
                'strategy_type': 'mean_reversion',
                'development_status': 'paper',
                'is_active': True,
                'is_starred': False,
                'backtesting_results': {
                    'total_trades': 200,
                    'winning_trades': 120,
                    'losing_trades': 80,
                    'total_return': 35.2,
                    'sharpe_ratio': 2.1,
                    'max_drawdown': 15.0,
                    'win_rate': 0.60,
                    'profit_factor': 1.95,
                    'calmar_ratio': 2.35
                }
            },
            {
                'name': 'MACD Momentum',
                'description': 'Momentum strategy using MACD and volume',
                'symbol': 'SOLUSDT',
                'timeframe': '1d',
                'strategy_type': 'momentum',
                'development_status': 'development',
                'is_active': False,
                'is_starred': True,
                'backtesting_results': {
                    'total_trades': 80,
                    'winning_trades': 45,
                    'losing_trades': 35,
                    'total_return': 42.8,
                    'sharpe_ratio': 2.4,
                    'max_drawdown': 18.2,
                    'win_rate': 0.56,
                    'profit_factor': 2.15,
                    'calmar_ratio': 2.35
                }
            }
        ]
        
        # Add strategies to database
        for strat_data in strategies:
            strategy = Strategy(
                name=strat_data['name'],
                description=strat_data['description'],
                symbol=strat_data['symbol'],
                timeframe=strat_data['timeframe'],
                strategy_type=strat_data['strategy_type'],
                development_status=strat_data['development_status'],
                is_active=strat_data['is_active'],
                is_starred=strat_data['is_starred'],
                backtesting_results=strat_data['backtesting_results'],
                total_return=strat_data['backtesting_results']['total_return'],
                sharpe_ratio=strat_data['backtesting_results']['sharpe_ratio'],
                win_rate=strat_data['backtesting_results']['win_rate'],
                max_drawdown=strat_data['backtesting_results']['max_drawdown'],
                profit_factor=strat_data['backtesting_results']['profit_factor'],
                calmar_ratio=strat_data['backtesting_results']['calmar_ratio']
            )
            db.session.add(strategy)
            db.session.flush()  # This assigns the ID to the strategy
            
            # Generate some sample trades for each strategy
            base_time = datetime.now() - timedelta(days=30)
            for i in range(10):
                entry_time = base_time + timedelta(hours=random.randint(0, 720))
                exit_time = entry_time + timedelta(hours=random.randint(1, 48))
                is_buy = random.choice([True, False])
                entry_price = random.uniform(20000, 30000) if strategy.symbol == 'BTCUSDT' else random.uniform(1500, 2000)
                
                # Generate realistic price movement
                price_change_pct = random.uniform(-0.05, 0.05)
                exit_price = entry_price * (1 + price_change_pct)
                
                trade = Trade(
                    strategy_id=strategy.id,  # Now strategy.id will be available
                    symbol=strategy.symbol,
                    side='BUY' if is_buy else 'SELL',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=random.uniform(0.1, 1.0),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    status='CLOSED',
                    timestamp=entry_time
                )
                trade.calculate_profit_loss()
                db.session.add(trade)
        
        db.session.commit()
        print("Test strategies and trades initialized successfully!")

if __name__ == "__main__":
    init_test_strategies()
