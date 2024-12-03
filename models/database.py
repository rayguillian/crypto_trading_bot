from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Strategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    symbol = db.Column(db.String(20), nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)
    strategy_type = db.Column(db.String(50), nullable=False)
    development_status = db.Column(db.String(20), default='development')
    is_active = db.Column(db.Boolean, default=False)
    is_starred = db.Column(db.Boolean, default=False)
    backtesting_results = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'strategy_type': self.strategy_type,
            'development_status': self.development_status,
            'is_active': self.is_active,
            'is_starred': self.is_starred,
            'backtesting_results': self.backtesting_results,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }

    # Performance metrics
    total_return = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    profit_factor = db.Column(db.Float)
    calmar_ratio = db.Column(db.Float)
    
    trades = db.relationship('Trade', backref='strategy', lazy=True)

class Trade(db.Model):
    """Trade model for storing trade information"""
    id = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    side = db.Column(db.String(10), nullable=False)  # BUY or SELL
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    quantity = db.Column(db.Float, nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime)
    status = db.Column(db.String(20), nullable=False, default='OPEN')  # OPEN or CLOSED
    profit_loss = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def calculate_profit_loss(self):
        if self.exit_price and self.status == 'CLOSED':
            if self.side == 'BUY':
                self.profit_loss = (self.exit_price - self.entry_price) * self.quantity
            else:
                self.profit_loss = (self.entry_price - self.exit_price) * self.quantity
