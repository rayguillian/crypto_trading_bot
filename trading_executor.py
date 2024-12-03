from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from decimal import Decimal, ROUND_DOWN
import os
from dotenv import load_dotenv
import json
from datetime import datetime

class TradingExecutor:
    """
    Handles both paper trading and live trading execution using the Binance API.
    """
    
    def __init__(self, api_key: str, api_secret: str, mode: str = 'paper'):
        """
        Initialize trading executor
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            mode: Trading mode ('paper' or 'live')
        """
        self.client = Client(api_key, api_secret)
        self.paper_trading = mode == 'paper'
        self.positions = {}  # Track open positions
        self.paper_balance = float(os.getenv('INITIAL_CAPITAL', '10000.0'))
        self.paper_positions: Dict[str, Dict] = {}
        self.trade_history: list = []
        
        # Load or create paper trading state file
        self.state_file = 'paper_trading_state.json'
        self.load_paper_trading_state()
        
    def load_paper_trading_state(self):
        """Load paper trading state from file if it exists."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.paper_balance = state.get('balance', self.paper_balance)
                    self.paper_positions = state.get('positions', {})
                    self.trade_history = state.get('trade_history', [])
        except Exception as e:
            logging.error(f"Error loading paper trading state: {e}")
            
    def save_paper_trading_state(self):
        """Save paper trading state to file."""
        try:
            state = {
                'balance': self.paper_balance,
                'positions': self.paper_positions,
                'trade_history': self.trade_history
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving paper trading state: {e}")
            
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get trading rules and precision for a symbol."""
        return self.client.get_symbol_info(symbol)
        
    def get_ticker_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
        
    def get_account_balance(self, asset: str = 'USDT') -> float:
        """
        Get account balance for an asset.
        
        Args:
            asset (str): Asset symbol (default: 'USDT')
            
        Returns:
            float: Balance amount
        """
        if self.paper_trading:
            return self.paper_balance
        else:
            account = self.client.get_account()
            balance = next((b for b in account['balances'] if b['asset'] == asset), None)
            return float(balance['free']) if balance else 0.0
            
    def format_quantity(self, symbol: str, quantity: float) -> float:
        """Format quantity according to symbol's precision rules."""
        info = self.get_symbol_info(symbol)
        step_size = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')['stepSize']
        precision = len(str(float(step_size)).rstrip('0').split('.')[-1])
        return float(Decimal(str(quantity)).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))
        
    def execute_order(self, 
                     symbol: str,
                     side: str,
                     quantity: float,
                     order_type: str = ORDER_TYPE_MARKET,
                     price: Optional[float] = None) -> Dict:
        """
        Execute a trade order (either paper or live).
        
        Args:
            symbol (str): Trading pair symbol
            side (str): Order side ('BUY' or 'SELL')
            quantity (float): Order quantity
            order_type (str): Order type (default: MARKET)
            price (float, optional): Limit price for limit orders
            
        Returns:
            Dict: Order information
        """
        quantity = self.format_quantity(symbol, quantity)
        current_price = price or self.get_ticker_price(symbol)
        
        if self.paper_trading:
            # Simulate paper trading
            order_value = quantity * current_price
            commission = order_value * 0.001  # Simulate 0.1% trading fee
            
            if side == 'BUY':
                if order_value + commission > self.paper_balance:
                    raise Exception("Insufficient paper trading balance")
                    
                self.paper_balance -= (order_value + commission)
                if symbol not in self.paper_positions:
                    self.paper_positions[symbol] = {'quantity': 0, 'entry_price': 0}
                    
                # Update position
                position = self.paper_positions[symbol]
                new_quantity = position['quantity'] + quantity
                position['entry_price'] = ((position['quantity'] * position['entry_price']) + 
                                         (quantity * current_price)) / new_quantity
                position['quantity'] = new_quantity
                
            else:  # SELL
                if symbol not in self.paper_positions or self.paper_positions[symbol]['quantity'] < quantity:
                    raise Exception("Insufficient paper trading position")
                    
                self.paper_balance += (order_value - commission)
                position = self.paper_positions[symbol]
                position['quantity'] -= quantity
                
                if position['quantity'] == 0:
                    del self.paper_positions[symbol]
                    
            # Record trade
            trade = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': current_price,
                'commission': commission,
                'timestamp': datetime.now().isoformat(),
                'type': 'PAPER'
            }
            self.trade_history.append(trade)
            self.save_paper_trading_state()
            
            return trade
            
        else:
            # Execute live trading
            try:
                if order_type == ORDER_TYPE_LIMIT and price:
                    order = self.client.create_order(
                        symbol=symbol,
                        side=side,
                        type=order_type,
                        timeInForce=TIME_IN_FORCE_GTC,
                        quantity=quantity,
                        price=price
                    )
                else:
                    order = self.client.create_order(
                        symbol=symbol,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=quantity
                    )
                    
                return order
                
            except Exception as e:
                logging.error(f"Error executing live order: {e}")
                raise
                
    def get_position(self, symbol: str) -> Dict[str, float]:
        """
        Get current position for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict[str, float]: Position information
        """
        if self.paper_trading:
            return self.paper_positions.get(symbol, {'quantity': 0, 'entry_price': 0})
        else:
            try:
                account = self.client.get_account()
                base_asset = symbol[:-4]  # Assuming USDT pairs
                balance = next((b for b in account['balances'] if b['asset'] == base_asset), None)
                quantity = float(balance['free']) if balance else 0
                
                # Get average entry price (approximate from recent trades)
                trades = self.client.get_my_trades(symbol=symbol, limit=100)
                if trades and quantity > 0:
                    # Calculate weighted average entry price from recent trades
                    total_qty = 0
                    total_value = 0
                    for trade in reversed(trades):
                        if total_qty >= quantity:
                            break
                        trade_qty = float(trade['qty'])
                        trade_price = float(trade['price'])
                        total_qty += trade_qty
                        total_value += trade_qty * trade_price
                    
                    entry_price = total_value / total_qty if total_qty > 0 else 0
                else:
                    entry_price = 0
                    
                return {'quantity': quantity, 'entry_price': entry_price}
                
            except Exception as e:
                logging.error(f"Error getting position: {e}")
                return {'quantity': 0, 'entry_price': 0}
                
    def get_trade_history(self, symbol: str = None) -> list:
        """
        Get trading history.
        
        Args:
            symbol (str, optional): Filter by symbol
            
        Returns:
            list: List of trades
        """
        if self.paper_trading:
            if symbol:
                return [t for t in self.trade_history if t['symbol'] == symbol]
            return self.trade_history
        else:
            try:
                if symbol:
                    return self.client.get_my_trades(symbol=symbol)
                
                # For live trading, we need to aggregate trades from all symbols
                trades = []
                symbols = [s['symbol'] for s in self.client.get_exchange_info()['symbols']]
                for sym in symbols:
                    try:
                        trades.extend(self.client.get_my_trades(symbol=sym))
                    except:
                        continue
                return trades
                
            except Exception as e:
                logging.error(f"Error getting trade history: {e}")
                return []

    def execute_signal(self, signal: dict, strategy_id: int):
        """Execute a trading signal"""
        try:
            if not signal or 'action' not in signal:
                logging.warning(f"Invalid signal received for strategy {strategy_id}")
                return
                
            symbol = signal.get('symbol')
            action = signal.get('action')
            price = signal.get('price')
            
            if not all([symbol, action, price]):
                logging.warning(f"Missing required signal parameters for strategy {strategy_id}")
                return
                
            # Calculate position size
            position_size = self.calculate_position_size(symbol, price)
            
            if action == 'BUY':
                if self.paper_trading:
                    self._paper_buy(symbol, position_size, price, strategy_id)
                else:
                    self._live_buy(symbol, position_size, price, strategy_id)
                    
            elif action == 'SELL':
                if self.paper_trading:
                    self._paper_sell(symbol, position_size, price, strategy_id)
                else:
                    self._live_sell(symbol, position_size, price, strategy_id)
                    
            logging.info(f"Successfully executed {action} signal for strategy {strategy_id}")
            
        except Exception as e:
            logging.error(f"Error executing signal for strategy {strategy_id}: {str(e)}")

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get account balance (paper or live)
            if self.paper_trading:
                balance = self.paper_balance
            else:
                balance = float(self.client.get_asset_balance(asset='USDT')['free'])
            
            # Risk 2% of account per trade
            risk_amount = balance * 0.02
            
            # Calculate quantity based on price
            quantity = risk_amount / price
            
            # Round to appropriate decimal places
            info = self.client.get_symbol_info(symbol)
            step_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', info['filters']))['stepSize']
            precision = len(str(float(step_size)).split('.')[-1])
            quantity = round(quantity, precision)
            
            return quantity
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _paper_buy(self, symbol: str, quantity: float, price: float, strategy_id: int):
        """Execute paper buy order"""
        try:
            cost = quantity * price
            
            if cost > self.paper_balance:
                logging.warning(f"Insufficient paper balance for buy order: {cost} > {self.paper_balance}")
                return
                
            # Create paper trade record
            trade = Trade(
                strategy_id=strategy_id,
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                price=price,
                timestamp=datetime.utcnow()
            )
            
            with self.app.app_context():
                db.session.add(trade)
                db.session.commit()
            
            # Update paper balance
            self.paper_balance -= cost
            self.paper_positions[symbol] = {
                'quantity': quantity,
                'entry_price': price
            }
            
            logging.info(f"Paper buy executed: {quantity} {symbol} @ {price}")
            
        except Exception as e:
            logging.error(f"Error executing paper buy: {str(e)}")

    def _paper_sell(self, symbol: str, quantity: float, price: float, strategy_id: int):
        """Execute paper sell order"""
        try:
            position = self.paper_positions.get(symbol)
            
            if not position or position['quantity'] < quantity:
                logging.warning(f"Insufficient paper position for sell order: {symbol}")
                return
                
            # Calculate profit/loss
            entry_price = position['entry_price']
            pnl = (price - entry_price) * quantity
            
            # Create paper trade record
            trade = Trade(
                strategy_id=strategy_id,
                symbol=symbol,
                side='SELL',
                quantity=quantity,
                price=price,
                pnl=pnl,
                timestamp=datetime.utcnow()
            )
            
            with self.app.app_context():
                db.session.add(trade)
                db.session.commit()
            
            # Update paper balance and position
            self.paper_balance += quantity * price
            
            remaining_quantity = position['quantity'] - quantity
            if remaining_quantity > 0:
                self.paper_positions[symbol]['quantity'] = remaining_quantity
            else:
                del self.paper_positions[symbol]
            
            logging.info(f"Paper sell executed: {quantity} {symbol} @ {price}, PnL: {pnl}")
            
        except Exception as e:
            logging.error(f"Error executing paper sell: {str(e)}")

    def _live_buy(self, symbol: str, quantity: float, price: float, strategy_id: int):
        """Execute live buy order"""
        try:
            # TO DO: implement live buy logic
            pass
            
        except Exception as e:
            logging.error(f"Error executing live buy: {str(e)}")

    def _live_sell(self, symbol: str, quantity: float, price: float, strategy_id: int):
        """Execute live sell order"""
        try:
            # TO DO: implement live sell logic
            pass
            
        except Exception as e:
            logging.error(f"Error executing live sell: {str(e)}")

class TradingExecutorNew:
    def __init__(self, api_key: str, api_secret: str, paper_trading: bool = True):
        """Initialize trading executor"""
        self.client = Client(api_key, api_secret)
        self.paper_trading = paper_trading
        self.positions = {}  # Track open positions
        
    def execute_signal(self, strategy_id: int, symbol: str, signal: int, 
                      risk_amount: float = 100.0) -> Optional[Dict]:
        """Execute a trading signal
        
        Args:
            strategy_id: ID of the strategy generating the signal
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            signal: Trading signal (1 for buy, -1 for sell, 0 for hold)
            risk_amount: Amount to risk per trade in USDT
            
        Returns:
            Dict containing trade details if executed, None if no trade
        """
        try:
            if signal == 0:  # No trade
                return None
                
            # Get current market price
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Calculate position size based on risk amount
            quantity = self._calculate_position_size(symbol, risk_amount, current_price)
            
            if signal == 1:  # Buy signal
                return self._execute_buy(strategy_id, symbol, quantity, current_price)
            elif signal == -1:  # Sell signal
                return self._execute_sell(strategy_id, symbol, quantity, current_price)
                
        except Exception as e:
            logging.error(f"Error executing signal for {symbol}: {str(e)}")
            return None
            
    def _execute_buy(self, strategy_id: int, symbol: str, quantity: float, 
                    price: float) -> Optional[Dict]:
        """Execute a buy order"""
        try:
            if self.paper_trading:
                # Paper trading - simulate order
                trade = Trade(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    side='BUY',
                    entry_price=price,
                    quantity=quantity,
                    entry_time=datetime.utcnow(),
                    status='OPEN'
                )
                db.session.add(trade)
                db.session.commit()
                
                # Track position
                self.positions[symbol] = {
                    'trade_id': trade.id,
                    'quantity': quantity,
                    'entry_price': price
                }
                
                return {
                    'trade_id': trade.id,
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'status': 'FILLED'
                }
                
            else:
                # Live trading - place actual order
                order = self.client.create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=quantity
                )
                
                # Save trade to database
                trade = Trade(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    side='BUY',
                    entry_price=float(order['fills'][0]['price']),
                    quantity=float(order['executedQty']),
                    entry_time=datetime.utcnow(),
                    status='OPEN'
                )
                db.session.add(trade)
                db.session.commit()
                
                return {
                    'trade_id': trade.id,
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': float(order['executedQty']),
                    'price': float(order['fills'][0]['price']),
                    'status': order['status']
                }
                
        except Exception as e:
            logging.error(f"Error executing buy order for {symbol}: {str(e)}")
            return None
            
    def _execute_sell(self, strategy_id: int, symbol: str, quantity: float, 
                     price: float) -> Optional[Dict]:
        """Execute a sell order"""
        try:
            if self.paper_trading:
                # Paper trading - simulate order
                # Check if we have an open position
                position = self.positions.get(symbol)
                if not position:
                    logging.warning(f"No open position found for {symbol}")
                    return None
                    
                # Close the position
                trade = Trade.query.get(position['trade_id'])
                if trade and trade.status == 'OPEN':
                    trade.exit_price = price
                    trade.exit_time = datetime.utcnow()
                    trade.status = 'CLOSED'
                    trade.calculate_profit_loss()
                    db.session.commit()
                    
                    # Remove position tracking
                    del self.positions[symbol]
                    
                    return {
                        'trade_id': trade.id,
                        'symbol': symbol,
                        'side': 'SELL',
                        'quantity': quantity,
                        'price': price,
                        'status': 'FILLED',
                        'profit_loss': trade.profit_loss
                    }
                    
            else:
                # Live trading - place actual order
                order = self.client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity
                )
                
                # Update trade in database
                position = self.positions.get(symbol)
                if position:
                    trade = Trade.query.get(position['trade_id'])
                    if trade and trade.status == 'OPEN':
                        trade.exit_price = float(order['fills'][0]['price'])
                        trade.exit_time = datetime.utcnow()
                        trade.status = 'CLOSED'
                        trade.calculate_profit_loss()
                        db.session.commit()
                        
                return {
                    'trade_id': trade.id if position else None,
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': float(order['executedQty']),
                    'price': float(order['fills'][0]['price']),
                    'status': order['status'],
                    'profit_loss': trade.profit_loss if position else None
                }
                
        except Exception as e:
            logging.error(f"Error executing sell order for {symbol}: {str(e)}")
            return None
            
    def _calculate_position_size(self, symbol: str, risk_amount: float, 
                               current_price: float) -> float:
        """Calculate position size based on risk amount"""
        try:
            # Get symbol info for quantity precision
            symbol_info = self.client.get_symbol_info(symbol)
            quantity_precision = 0
            
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    quantity_precision = len(str(float(filter['stepSize'])).rstrip('0').split('.')[1])
                    break
                    
            # Calculate quantity
            quantity = risk_amount / current_price
            
            # Round to appropriate precision
            quantity = float(Decimal(str(quantity)).quantize(
                Decimal('0.' + '0' * quantity_precision),
                rounding=ROUND_DOWN
            ))
            
            return quantity
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0.0
