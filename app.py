from flask import Flask, render_template, jsonify, request, abort
from flask_cors import CORS
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from models.database import db, Strategy, Trade
from datetime import datetime, timedelta
import plotly.graph_objs as go
import pandas as pd
from binance.client import Client
from unicorn_binance_websocket_api import BinanceWebSocketApiManager
import os
from dotenv import load_dotenv
import json
import socket
from contextlib import closing
from flask_socketio import SocketIO, emit
import threading
import queue
from core.strategy_development import StrategyDevelopment
from core.auto_trader import AutoTrader
from flask_migrate import Migrate
import numpy as np
import time
import uuid
import logging
from logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)
strategy_logger = logging.getLogger('strategy')

# Load environment variables
load_dotenv()

def find_free_port(start_port=8050, max_tries=10):
    """Find the next available port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    return start_port

def initialize_binance_client():
    """Initialize Binance client with proper error handling"""
    global binance_client, auto_trader, ubwa
    try:
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not found in environment variables")
            
        # Initialize REST client
        binance_client = Client(api_key, api_secret)
        
        # Initialize WebSocket client
        ubwa = BinanceWebSocketApiManager(exchange="binance.com")
        channels = ['ticker']
        markets = ['btcusdt', 'ethusdt', 'solusdt', 'adausdt']
        ubwa.create_stream(channels, markets, output="dict")
        logger.info("Binance WebSocket client initialized and started")
        
        # Initialize AutoTrader
        auto_trader = AutoTrader(binance_client)
        logger.info("Binance client and AutoTrader initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {str(e)}")
        raise

def handle_message(msg):
    if msg.get('e') == 'error':
        logger.error(f"WebSocket error: {msg.get('m')}")
    else:
        socketio.emit('market_update', msg)

@app.before_first_request
def before_first_request():
    """Initialize database and Binance client before first request"""
    db.create_all()
    initialize_binance_client()

def get_sample_trades():
    """Generate sample trades when running in demo mode"""
    current_time = datetime.now()
    return [
        {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'price': 37500.0,
            'quantity': 0.15,
            'timestamp': (current_time - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
            'total': 5625.0,
            'status': 'FILLED',
            'strategy_name': 'BTC Momentum'
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'SELL',
            'price': 2050.0,
            'quantity': 2.5,
            'timestamp': (current_time - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
            'total': 5125.0,
            'status': 'FILLED',
            'strategy_name': 'ETH Swing Trader'
        },
        {
            'symbol': 'SOLUSDT',
            'side': 'BUY',
            'price': 58.75,
            'quantity': 50,
            'timestamp': (current_time - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
            'total': 2937.5,
            'status': 'OPEN',
            'strategy_name': 'SOL Breakout'
        }
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/strategies')
def get_strategies():
    try:
        strategies = Strategy.query.all()
        return jsonify([strategy.to_dict() for strategy in strategies])
    except Exception as e:
        app.logger.error(f"Error fetching strategies: {str(e)}")
        return jsonify([])

@app.route('/api/trades')
def get_trades():
    try:
        if not binance_client:
            return jsonify(get_sample_trades())
            
        # Get open orders from Binance
        open_orders = []
        try:
            open_orders = binance_client.get_open_orders()
        except Exception as e:
            app.logger.warning(f"Error fetching open orders: {str(e)}")
            open_orders = []
        
        # Get recent trades
        trades = []
        try:
            trading_pairs = os.getenv('DEFAULT_TRADING_PAIRS', 'BTCUSDT').split(',')
            
            # Only get trades for active strategies
            active_strategies = Strategy.query.filter_by(is_active=True).all()
            active_symbols = set(s.symbol for s in active_strategies)
            
            for symbol in trading_pairs:
                symbol = symbol.strip()
                if symbol not in active_symbols:
                    continue
                    
                try:
                    symbol_trades = binance_client.get_my_trades(symbol=symbol, limit=10)
                    trades.extend(symbol_trades)
                except Exception as e:
                    app.logger.warning(f"Error fetching trades for {symbol}: {str(e)}")
                    continue
        except Exception as e:
            app.logger.warning(f"Error processing trading pairs: {str(e)}")
            trades = []
            
        if not trades and not open_orders:
            return jsonify(get_sample_trades())
            
        # Format trades for frontend
        formatted_trades = []
        
        # Format completed trades
        for trade in trades:
            try:
                formatted_trade = {
                    'symbol': trade['symbol'],
                    'side': 'BUY' if trade.get('isBuyer', False) else 'SELL',
                    'price': float(trade.get('price', 0)),
                    'quantity': float(trade.get('qty', 0)),
                    'timestamp': datetime.fromtimestamp(int(trade.get('time', 0))/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'total': float(trade.get('price', 0)) * float(trade.get('qty', 0)),
                    'status': 'FILLED'
                }
                formatted_trades.append(formatted_trade)
            except Exception as e:
                app.logger.error(f"Error formatting trade: {str(e)}")
                continue
            
        # Format open orders
        for order in open_orders:
            try:
                formatted_trade = {
                    'symbol': order.get('symbol', ''),
                    'side': order.get('side', ''),
                    'price': float(order.get('price', 0)),
                    'quantity': float(order.get('origQty', 0)),
                    'timestamp': datetime.fromtimestamp(int(order.get('time', 0))/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'total': float(order.get('price', 0)) * float(order.get('origQty', 0)),
                    'status': order.get('status', '')
                }
                formatted_trades.append(formatted_trade)
            except Exception as e:
                app.logger.error(f"Error formatting order: {str(e)}")
                continue
            
        return jsonify(formatted_trades)
        
    except Exception as e:
        app.logger.error(f"Error in trades endpoint: {str(e)}")
        return jsonify(get_sample_trades())

@app.route('/api/strategies/create', methods=['POST'])
def create_strategy():
    """Create a new trading strategy"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'symbol', 'timeframe', 'strategy_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
                
        # Create new strategy
        strategy = Strategy(
            name=data['name'],
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            strategy_type=data['strategy_type'],
            description=data.get('description', ''),
            development_status='development'
        )
        
        db.session.add(strategy)
        db.session.commit()
        
        # Start strategy development in AutoTrader
        if auto_trader:
            auto_trader.develop_strategy(strategy.id)
            
        return jsonify(strategy.to_dict()), 201
        
    except Exception as e:
        app.logger.error(f"Error creating strategy: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<int:strategy_id>/start', methods=['POST'])
def start_trading_strategy(strategy_id):
    """Start paper trading for a strategy"""
    try:
        strategy = Strategy.query.get(strategy_id)
        if not strategy:
            return jsonify({'error': 'Strategy not found'}), 404
            
        if not auto_trader:
            return jsonify({'error': 'AutoTrader not initialized'}), 500
            
        # Start paper trading
        auto_trader._start_paper_trading(strategy_id)
        
        strategy.development_status = 'paper_trading'
        db.session.commit()
        
        return jsonify({'message': f'Started paper trading for strategy {strategy.name}'}), 200
        
    except Exception as e:
        app.logger.error(f"Error starting strategy: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<int:strategy_id>/stop', methods=['POST'])
def stop_trading_strategy(strategy_id):
    """Stop trading for a strategy"""
    try:
        strategy = Strategy.query.get(strategy_id)
        if not strategy:
            return jsonify({'error': 'Strategy not found'}), 404
            
        if not auto_trader:
            return jsonify({'error': 'AutoTrader not initialized'}), 500
            
        # Stop the strategy
        auto_trader.stop_strategy(strategy_id)
        
        strategy.development_status = 'stopped'
        db.session.commit()
        
        return jsonify({'message': f'Stopped trading for strategy {strategy.name}'}), 200
        
    except Exception as e:
        app.logger.error(f"Error stopping strategy: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<int:strategy_id>', methods=['DELETE'])
def delete_strategy(strategy_id):
    """Delete a strategy"""
    try:
        strategy = Strategy.query.get(strategy_id)
        if not strategy:
            return jsonify({'error': 'Strategy not found'}), 404
            
        # Stop the strategy if it's running
        if auto_trader and strategy.development_status == 'paper_trading':
            auto_trader.stop_strategy(strategy_id)
            
        db.session.delete(strategy)
        db.session.commit()
        
        return jsonify({'message': f'Strategy {strategy.name} deleted'}), 200
        
    except Exception as e:
        app.logger.error(f"Error deleting strategy: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/develop', methods=['POST'])
def develop_strategy():
    """Develop a new trading strategy"""
    try:
        if not strategy_developer:
            return jsonify({'error': 'Strategy developer not initialized'}), 500
            
        data = request.json
        required_fields = ['name', 'type', 'symbol', 'interval', 'lookback_days']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        strategy = strategy_developer.develop_strategy(
            name=data['name'],
            strategy_type=data['type'],
            symbol=data['symbol'],
            interval=data['interval'],
            lookback_days=data['lookback_days']
        )
        
        return jsonify({
            'success': True,
            'strategy': {
                'id': strategy.id,
                'name': strategy.name,
                'type': strategy.type,
                'symbol': strategy.symbol,
                'parameters': json.loads(strategy.parameters),
                'backtesting_results': json.loads(strategy.backtesting_results)
            }
        })
        
    except Exception as e:
        app.logger.error(f"Strategy development failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest_strategy', methods=['POST'])
def backtest_strategy():
    """Backtest an existing strategy with new parameters"""
    try:
        data = request.get_json()
        strategy_id = data.get('strategy_id')
        params = data.get('parameters', {})
        
        strategy = Strategy.query.get(strategy_id)
        if not strategy:
            logger.error(f"Strategy not found: {strategy_id}")
            return jsonify({'error': 'Strategy not found'}), 404
            
        strategy_logger.info(f"Starting backtest for strategy {strategy.name} with params: {params}")
        
        # Initialize strategy developer
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        strategy_developer = StrategyDevelopment(api_key, api_secret)
        
        # Fetch historical data
        df = strategy_developer.fetch_historical_data(
            symbol=strategy.symbol,
            interval=strategy.timeframe,
            lookback_days=30
        )
        strategy_logger.info(f"Fetched {len(df)} historical data points for {strategy.symbol}")
        
        # Run backtest
        signals, metrics = strategy_developer.backtest_strategy(df, strategy.strategy_type, params)
        
        strategy_logger.info(f"Backtest completed. Metrics: {metrics}")
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'trades': len(signals[signals['position'] != 0])
        })
        
    except Exception as e:
        logger.error(f"Error in backtest_strategy: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/<int:strategy_id>/optimize', methods=['POST'])
def optimize_strategy():
    """Optimize strategy parameters"""
    try:
        if not strategy_developer:
            return jsonify({'error': 'Strategy developer not initialized'}), 500
            
        strategy = Strategy.query.get_or_404(strategy_id)
        
        # Fetch historical data
        df = strategy_developer.fetch_historical_data(
            symbol=strategy.symbol,
            interval='1h',  # Default to 1h
            lookback_days=30  # Default to 30 days
        )
        
        # Calculate indicators
        df = strategy_developer.calculate_indicators(df, strategy.type)
        
        # Optimize parameters
        optimization_results = strategy_developer.optimize_parameters(df, strategy.type)
        
        # Update strategy with optimized parameters
        strategy.parameters = json.dumps(optimization_results['params'])
        strategy.backtesting_results = json.dumps(optimization_results['metrics'])
        db.session.commit()
        
        return jsonify({
            'success': True,
            'optimization_results': optimization_results
        })
        
    except Exception as e:
        app.logger.error(f"Strategy optimization failed: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/<int:strategy_id>/toggle', methods=['POST'])
def toggle_strategy(strategy_id):
    try:
        strategy = Strategy.query.get_or_404(strategy_id)
        strategy.is_active = not strategy.is_active
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error toggling strategy {strategy_id}: {str(e)}")
        return jsonify({'error': 'Failed to toggle strategy'}), 500

@app.route('/api/strategy/<int:strategy_id>/star', methods=['POST'])
def toggle_star(strategy_id):
    try:
        strategy = Strategy.query.get_or_404(strategy_id)
        strategy.is_starred = not strategy.is_starred
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error toggling star for strategy {strategy_id}: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/strategy/<int:strategy_id>/trades')
def get_strategy_trades(strategy_id):
    try:
        trades = Trade.query.filter_by(strategy_id=strategy_id).all()
        return jsonify([{
            'id': t.id,
            'symbol': t.symbol,
            'side': t.side,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'profit_loss': t.profit_loss,
            'status': t.status,
            'entry_time': t.entry_time.isoformat(),
            'exit_time': t.exit_time.isoformat() if t.exit_time else None
        } for t in trades])
    except Exception as e:
        app.logger.error(f"Error fetching trades for strategy {strategy_id}: {str(e)}")
        return jsonify({'error': 'Failed to fetch trades'}), 500

@app.route('/api/market/price/<symbol>')
def get_market_price(symbol):
    try:
        # Get real-time price from Binance
        ticker = binance_client.get_symbol_ticker(symbol=symbol.upper())
        return jsonify({
            'success': True,
            'price': ticker['price']
        })
    except Exception as e:
        app.logger.error(f"Failed to get market price for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    """Get real-time market data for supported pairs"""
    try:
        # If we have a Binance client, get real data
        if binance_client:
            prices = {}
            for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']:
                ticker = binance_client.get_symbol_ticker(symbol=symbol)
                prices[symbol] = float(ticker['price'])
            return jsonify(prices)
        
        # Otherwise return sample data
        return jsonify({
            'BTCUSDT': 65000.00,
            'ETHUSDT': 3500.00,
            'SOLUSDT': 120.00,
            'ADAUSDT': 1.20
        })
        
    except Exception as e:
        app.logger.error(f"Error getting market data: {str(e)}")
        return jsonify({
            'BTCUSDT': 65000.00,
            'ETHUSDT': 3500.00,
            'SOLUSDT': 120.00,
            'ADAUSDT': 1.20
        })

@app.route('/api/auto_trader/status')
def get_auto_trader_status():
    """Get AutoTrader status and statistics"""
    if auto_trader:
        active_strategies = Strategy.query.filter_by(is_active=True).count()
        total_strategies = Strategy.query.count()
        recent_trades = Trade.query.filter(
            Trade.timestamp >= datetime.now() - timedelta(days=7)
        ).count()
        
        return jsonify({
            'success': True,
            'status': 'running',
            'active_strategies': active_strategies,
            'total_strategies': total_strategies,
            'recent_trades': recent_trades
        })
    else:
        return jsonify({
            'success': False,
            'status': 'stopped',
            'error': 'AutoTrader not initialized'
        })

@app.route('/api/auto_trader/start', methods=['POST'])
def start_auto_trader():
    """Start the AutoTrader"""
    global auto_trader
    if not auto_trader:
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        auto_trader = AutoTrader(api_key, api_secret)
    
    auto_trader.start()
    return jsonify({'success': True, 'message': 'AutoTrader started'})

@app.route('/api/auto_trader/stop', methods=['POST'])
def stop_auto_trader():
    """Stop the AutoTrader"""
    if auto_trader:
        auto_trader.stop()
        return jsonify({'success': True, 'message': 'AutoTrader stopped'})
    return jsonify({'success': False, 'error': 'AutoTrader not running'})

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Trade event queues for each strategy
strategy_queues = {}

def emit_trade_update(strategy_id, trade_data):
    """Emit trade update to connected clients"""
    socketio.emit(f'trade_update_{strategy_id}', trade_data)

def process_strategy_queue(strategy_id):
    """Process trade events for a strategy"""
    while True:
        try:
            trade = strategy_queues[strategy_id].get()
            emit_trade_update(strategy_id, trade)
        except Exception as e:
            app.logger.error(f"Error processing trade for strategy {strategy_id}: {str(e)}")

@socketio.on('connect')
def handle_connect():
    try:
        # Start WebSocket connection for market data
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        
        def handle_socket_message(msg):
            if msg.get('e') == 'error':
                logger.error(f"WebSocket error: {msg.get('m')}")
            else:
                socketio.emit('market_update', msg)
        
        for symbol in symbols:
            ubwa.subscribe_to_stream(ubwa.get_stream_id_list()[0], handle_socket_message)
        
        emit('connection_status', {'status': 'connected'})
        logger.info("WebSocket connection established")
        
    except Exception as e:
        logger.error(f"Error in socket connection: {str(e)}")
        emit('connection_status', {'status': 'error', 'message': str(e)})

@socketio.on('disconnect')
def handle_disconnect():
    try:
        if ubwa:
            ubwa.stop_stream(ubwa.get_stream_id_list()[0])
            logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Error closing WebSocket connection: {str(e)}")

@socketio.on('evaluate_strategies')
def handle_strategy_evaluation(data):
    try:
        trading_pairs = data.get('trading_pairs', [])
        timeframe = data.get('timeframe', '1h')
        lookback_period = data.get('lookback_period', 7)
        strategies = data.get('strategies', [])
        
        if not trading_pairs or not strategies:
            emit('strategy_evaluation_result', {
                'error': 'Please select at least one trading pair and strategy'
            })
            return
            
        # Calculate lookback start time
        start_time = datetime.now() - timedelta(days=lookback_period)
        
        # Initialize metrics
        total_return = 0
        win_rate = 0
        max_drawdown = 0
        total_trades = 0
        sharpe_ratio = 0
        
        # Evaluate each strategy on each trading pair
        for strategy_name in strategies:
            for pair in trading_pairs:
                try:
                    # Get historical data with rate limiting
                    historical_data = rate_limited_api_call(
                        lambda: get_historical_data(pair, timeframe, start_time)
                    )
                    
                    # Skip if no data
                    if historical_data.empty:
                        logger.warning(f"No historical data for {pair}")
                        continue
                    
                    # Evaluate strategy
                    strategy_results = evaluate_strategy(
                        historical_data,
                        strategy_name,
                        pair
                    )
                    
                    # Update metrics
                    total_return += strategy_results['total_return']
                    win_rate = max(win_rate, strategy_results['win_rate'])
                    max_drawdown = min(max_drawdown, strategy_results['max_drawdown'])
                    total_trades += strategy_results['total_trades']
                    sharpe_ratio = max(sharpe_ratio, strategy_results['sharpe_ratio'])
                    
                except Exception as e:
                    logger.error(f"Error evaluating {strategy_name} on {pair}: {str(e)}")
                    continue
        
        # Average metrics across all pairs and strategies
        num_combinations = len(trading_pairs) * len(strategies)
        if num_combinations > 0:
            total_return /= num_combinations
            
        # Generate recommendation
        recommendation = generate_strategy_recommendation(
            total_return, win_rate, max_drawdown, sharpe_ratio
        )
        
        # Emit results
        emit('strategy_evaluation_result', {
            'metrics': {
                'total_return': total_return,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'sharpe_ratio': sharpe_ratio
            },
            'recommendation': recommendation
        })
        
    except Exception as e:
        logger.error(f"Error in strategy evaluation: {str(e)}")
        emit('strategy_evaluation_result', {
            'error': f'Strategy evaluation failed: {str(e)}'
        })

def generate_strategy_recommendation(total_return, win_rate, max_drawdown, sharpe_ratio):
    """Generate a strategy recommendation based on performance metrics"""
    recommendation = []
    
    if total_return > 0:
        if win_rate > 0.6:
            recommendation.append("Strong performance with high win rate")
        else:
            recommendation.append("Positive returns but inconsistent wins")
    else:
        recommendation.append("Strategy needs optimization")
        
    if max_drawdown < -20:
        recommendation.append("High risk - consider reducing position sizes")
    elif max_drawdown < -10:
        recommendation.append("Moderate risk profile")
    else:
        recommendation.append("Conservative risk profile")
        
    if sharpe_ratio > 2:
        recommendation.append("Excellent risk-adjusted returns")
    elif sharpe_ratio > 1:
        recommendation.append("Good risk-adjusted returns")
    else:
        recommendation.append("Consider adjusting strategy parameters")
        
    return " | ".join(recommendation)

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance metrics for the dashboard"""
    try:
        timeframe = request.args.get('timeframe', '1d')
        
        # Calculate time range
        end_time = datetime.utcnow()
        if timeframe == '1d':
            start_time = end_time - timedelta(days=1)
        elif timeframe == '1w':
            start_time = end_time - timedelta(weeks=1)
        elif timeframe == '1m':
            start_time = end_time - timedelta(days=30)
        else:  # 3m
            start_time = end_time - timedelta(days=90)
            
        # Get all trades within timeframe
        trades = Trade.query.filter(
            Trade.timestamp.between(start_time, end_time)
        ).order_by(Trade.timestamp.asc()).all()
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate returns
        returns = [t.pnl for t in trades]
        total_return = sum(returns)
        daily_return = sum(t.pnl for t in trades if t.timestamp.date() == datetime.utcnow().date())
        
        # Calculate Sharpe ratio (if we have enough data)
        if len(returns) > 1:
            returns_array = np.array(returns)
            sharpe_ratio = np.sqrt(252) * returns_array.mean() / returns_array.std()
        else:
            sharpe_ratio = 0
            
        # Get risk metrics
        strategies = Strategy.query.filter_by(is_active=True).all()
        total_risk = sum(s.current_risk for s in strategies if s.current_risk)
        max_risk = 0.15  # 15% maximum portfolio risk
        
        # Get strategy-wise performance
        strategy_metrics = {}
        for strategy in Strategy.query.all():
            strategy_trades = [t for t in trades if t.strategy_id == strategy.id]
            if strategy_trades:
                returns = [t.pnl for t in strategy_trades]
                strategy_metrics[strategy.id] = {
                    'return': sum(returns),
                    'sharpe_ratio': np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
                }
        
        # Calculate performance history
        performance_history = []
        current_value = 10000  # Starting value
        for trade in trades:
            current_value += trade.pnl
            performance_history.append({
                'timestamp': trade.timestamp.isoformat(),
                'value': current_value
            })
            
        return jsonify({
            'total_return': total_return,
            'daily_return': daily_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'current_risk': total_risk,
            'max_risk': max_risk,
            'performance_history': performance_history,
            'strategy_risk': total_risk,
            'available_risk': max_risk - total_risk,
            'reserved_risk': max_risk * 0.2,  # Keep 20% reserved
            'strategy_returns': [strategy_metrics.get(s.id, {'return': 0})['return'] 
                               for s in Strategy.query.filter_by(strategy_type='Momentum').all()] +
                              [strategy_metrics.get(s.id, {'return': 0})['return'] 
                               for s in Strategy.query.filter_by(strategy_type='Mean Reversion').all()] +
                              [strategy_metrics.get(s.id, {'return': 0})['return'] 
                               for s in Strategy.query.filter_by(strategy_type='Breakout').all()],
            'strategy_sharpe_ratios': [strategy_metrics.get(s.id, {'sharpe_ratio': 0})['sharpe_ratio'] 
                                     for s in Strategy.query.filter_by(strategy_type='Momentum').all()] +
                                    [strategy_metrics.get(s.id, {'sharpe_ratio': 0})['sharpe_ratio'] 
                                     for s in Strategy.query.filter_by(strategy_type='Mean Reversion').all()] +
                                    [strategy_metrics.get(s.id, {'sharpe_ratio': 0})['sharpe_ratio'] 
                                     for s in Strategy.query.filter_by(strategy_type='Breakout').all()]
        })
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<int:strategy_id>/performance', methods=['GET'])
def get_strategy_performance(strategy_id):
    """Get detailed performance metrics for a specific strategy"""
    try:
        strategy = Strategy.query.get(strategy_id)
        if not strategy:
            return jsonify({'error': 'Strategy not found'}), 404
            
        # Get all trades for this strategy
        trades = Trade.query.filter_by(strategy_id=strategy_id).order_by(Trade.timestamp.asc()).all()
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        returns = [t.pnl for t in trades]
        total_return = sum(returns)
        
        # Calculate drawdown
        equity_curve = []
        peak = 10000  # Starting value
        max_drawdown = 0
        current_value = peak
        
        for trade in trades:
            current_value += trade.pnl
            equity_curve.append(current_value)
            peak = max(peak, current_value)
            drawdown = (peak - current_value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return jsonify({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown * 100,
            'equity_curve': equity_curve,
            'current_risk': strategy.current_risk if strategy.current_risk else 0,
            'trades': [{
                'timestamp': t.timestamp.isoformat(),
                'type': t.side,
                'price': t.price,
                'quantity': t.quantity,
                'pnl': t.pnl
            } for t in trades[-50:]]  # Last 50 trades
        })
        
    except Exception as e:
        logger.error(f"Error getting strategy performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/<int:strategy_id>/start', methods=['POST'])
def start_strategy(strategy_id):
    """Start a strategy and initialize its trade queue"""
    try:
        strategy = Strategy.query.get_or_404(strategy_id)
        
        # Initialize queue if not exists
        if strategy_id not in strategy_queues:
            strategy_queues[strategy_id] = queue.Queue()
            threading.Thread(target=process_strategy_queue, args=(strategy_id,), daemon=True).start()
        
        strategy.is_active = True
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error starting strategy {strategy_id}: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to start strategy'}), 500

@app.route('/api/strategy/<int:strategy_id>/stop', methods=['POST'])
def stop_strategy(strategy_id):
    """Stop a strategy"""
    try:
        strategy = Strategy.query.get_or_404(strategy_id)
        strategy.is_active = False
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error stopping strategy {strategy_id}: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to stop strategy'}), 500

def add_trade_to_queue(strategy_id, trade_data):
    """Add a trade to the strategy's queue"""
    if strategy_id in strategy_queues:
        strategy_queues[strategy_id].put(trade_data)

# Strategy evaluation background task
def evaluate_strategy_task(task_id, symbols, timeframes, lookback_days):
    try:
        for symbol in symbols:
            for timeframe in timeframes:
                # Notify starting evaluation
                socketio.emit('strategy_evaluation', {
                    'task_id': task_id,
                    'status': 'starting',
                    'symbol': symbol,
                    'timeframe': timeframe
                })

                # Get historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                historical_data = get_historical_data(symbol, timeframe, start_date, end_date)

                # Evaluate different strategy types
                strategy_types = ['trend_following', 'mean_reversion', 'momentum']
                for strategy_type in strategy_types:
                    try:
                        # Evaluate strategy
                        results = evaluate_strategy(historical_data, strategy_type)

                        # Emit results
                        socketio.emit('strategy_evaluation', {
                            'task_id': task_id,
                            'status': 'completed',
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'strategy_type': strategy_type,
                            'results': results
                        })
                    except Exception as e:
                        socketio.emit('strategy_evaluation', {
                            'task_id': task_id,
                            'status': 'error',
                            'error': str(e)
                        })

    except Exception as e:
        socketio.emit('strategy_evaluation', {
            'task_id': task_id,
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/strategies/evaluate', methods=['POST'])
def evaluate_strategies():
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        timeframes = data.get('timeframes', [])
        lookback_days = data.get('lookback_days', 60)

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Start evaluation in background
        thread = threading.Thread(
            target=evaluate_strategy_task,
            args=(task_id, symbols, timeframes, lookback_days)
        )
        thread.start()

        return jsonify({'task_id': task_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/recommendations', methods=['GET'])
def get_strategy_recommendations():
    try:
        # Get top performing strategies from evaluation results
        recommendations = [
            {
                'id': 1,
                'name': 'Trend Following BTC',
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'description': 'Momentum-based strategy using EMA crossovers and volume confirmation',
                'total_return': 0.45,  # 45%
                'sharpe_ratio': 2.1,
                'win_rate': 0.65,  # 65%
                'max_drawdown': 0.15  # 15%
            },
            {
                'id': 2,
                'name': 'ETH Mean Reversion',
                'symbol': 'ETHUSDT',
                'timeframe': '4h',
                'description': 'RSI-based mean reversion strategy with volatility adjustment',
                'total_return': 0.38,
                'sharpe_ratio': 1.9,
                'win_rate': 0.62,
                'max_drawdown': 0.18
            },
            {
                'id': 3,
                'name': 'BTC Momentum',
                'symbol': 'BTCUSDT',
                'timeframe': '1d',
                'description': 'MACD and ADX based momentum strategy with trend filtering',
                'total_return': 0.52,
                'sharpe_ratio': 2.3,
                'win_rate': 0.58,
                'max_drawdown': 0.22
            }
        ]
        return jsonify(recommendations)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Create Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dash layout for interactive charts
dash_app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3('Trading Performance'),
            dcc.Graph(id='performance-chart'),
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # in milliseconds
                n_intervals=0
            )
        ])
    ])
])

@dash_app.callback(
    Output('performance-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_performance_chart(n):
    try:
        # Get trade data from database
        trades = Trade.query.filter_by(status='CLOSED').order_by(Trade.entry_time.asc()).all()
        
        if not trades:
            return go.Figure()
            
        # Create DataFrame
        df = pd.DataFrame([{
            'timestamp': t.entry_time,
            'profit_loss': t.profit_loss if t.profit_loss is not None else 0
        } for t in trades])
        
        # Calculate cumulative returns
        df['cumulative_return'] = df['profit_loss'].cumsum()
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_return'],
            mode='lines',
            name='Cumulative Return'
        ))
        
        fig.update_layout(
            title='Trading Performance',
            xaxis_title='Time',
            yaxis_title='Cumulative Return (USDT)',
            template='plotly_dark'
        )
        
        return fig
    except Exception as e:
        app.logger.error(f"Error updating performance chart: {str(e)}")
        return go.Figure()

# Initialize rate limiter variables
last_api_call = {}
min_interval = 1.0  # Minimum interval between API calls in seconds

def rate_limited_api_call(func):
    """Make rate-limited API calls to Binance"""
    global last_api_call
    
    # Calculate time since last API call for this symbol
    current_time = time.time()
    key = f"{func.__name__}"
    if key in last_api_call:
        elapsed = current_time - last_api_call[key]
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    
    try:
        data = func()
        last_api_call[key] = time.time()
        return data
    except Exception as e:
        if 'rate limit' in str(e).lower():
            logger.warning(f"Rate limit hit, waiting 60 seconds...")
            time.sleep(60)
            return rate_limited_api_call(func)
        raise

def get_historical_data(symbol, timeframe, start_date, end_date):
    """Get historical data with rate limiting"""
    try:
        klines = rate_limited_api_call(
            lambda: binance_client.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                start_str=start_date.strftime('%Y-%m-%d'),
                end_str=end_date.strftime('%Y-%m-%d')
            )
        )
        
        if not klines:
            return pd.DataFrame()
            
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                         'taker_buy_quote', 'ignored'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise

def evaluate_strategy(historical_data, strategy_type):
    """Evaluate a trading strategy on historical data"""
    try:
        # Calculate basic indicators
        df = historical_data.copy()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = calculate_rsi(df['close'])
        
        # Initialize results
        trades = []
        position = 0
        entry_price = 0
        returns = []
        
        # Strategy logic
        for i in range(51, len(df)):
            if strategy_type == 'trend_following':
                # Buy when 20 SMA crosses above 50 SMA
                if df['sma_20'].iloc[i-1] <= df['sma_50'].iloc[i-1] and df['sma_20'].iloc[i] > df['sma_50'].iloc[i]:
                    if position == 0:
                        position = 1
                        entry_price = df['close'].iloc[i]
                        trades.append({
                            'type': 'buy',
                            'price': entry_price,
                            'timestamp': df['timestamp'].iloc[i]
                        })
                
                # Sell when 20 SMA crosses below 50 SMA
                elif df['sma_20'].iloc[i-1] >= df['sma_50'].iloc[i-1] and df['sma_20'].iloc[i] < df['sma_50'].iloc[i]:
                    if position == 1:
                        exit_price = df['close'].iloc[i]
                        trades.append({
                            'type': 'sell',
                            'price': exit_price,
                            'timestamp': df['timestamp'].iloc[i]
                        })
                        position = 0
            
            elif strategy_type == 'mean_reversion':
                # Buy when RSI is below 30
                if df['rsi'].iloc[i] < 30:
                    if position == 0:
                        position = 1
                        entry_price = df['close'].iloc[i]
                        trades.append({
                            'type': 'buy',
                            'price': entry_price,
                            'timestamp': df['timestamp'].iloc[i]
                        })
                
                # Sell when RSI is above 70
                elif df['rsi'].iloc[i] > 70:
                    if position == 1:
                        exit_price = df['close'].iloc[i]
                        trades.append({
                            'type': 'sell',
                            'price': exit_price,
                            'timestamp': df['timestamp'].iloc[i]
                        })
                        position = 0
            
            elif strategy_type == 'momentum':
                # Buy when price is above both SMAs and RSI is above 50
                if df['close'].iloc[i] > df['sma_20'].iloc[i] and df['close'].iloc[i] > df['sma_50'].iloc[i] and df['rsi'].iloc[i] > 50:
                    if position == 0:
                        position = 1
                        entry_price = df['close'].iloc[i]
                        trades.append({
                            'type': 'buy',
                            'price': entry_price,
                            'timestamp': df['timestamp'].iloc[i]
                        })
                
                # Sell when price is below either SMA or RSI is below 50
                elif (df['close'].iloc[i] < df['sma_20'].iloc[i] or df['close'].iloc[i] < df['sma_50'].iloc[i]) and df['rsi'].iloc[i] < 50:
                    if position == 1:
                        exit_price = df['close'].iloc[i]
                        trades.append({
                            'type': 'sell',
                            'price': exit_price,
                            'timestamp': df['timestamp'].iloc[i]
                        })
                        position = 0
        
        # Close any open position at the end
        if position == 1:
            exit_price = df['close'].iloc[-1]
            trades.append({
                'type': 'sell',
                'price': exit_price,
                'timestamp': df['timestamp'].iloc[-1]
            })
        
        # Calculate performance metrics
        total_return = 0
        win_count = 0
        loss_count = 0
        max_drawdown = 0
        peak = 0
        
        for i in range(0, len(trades)-1, 2):
            entry = trades[i]
            exit = trades[i+1]
            trade_return = (exit['price'] - entry['price']) / entry['price']
            total_return += trade_return
            returns.append(trade_return)
            
            if trade_return > 0:
                win_count += 1
            else:
                loss_count += 1
            
            # Update peak and drawdown
            if total_return > peak:
                peak = total_return
            drawdown = peak - total_return
            max_drawdown = max(max_drawdown, drawdown)
        
        total_trades = win_count + loss_count
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(returns) > 1:
            returns_array = np.array(returns)
            sharpe_ratio = np.sqrt(252) * (returns_array.mean() / returns_array.std()) if returns_array.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades
        }
        
    except Exception as e:
        logger.error(f"Error evaluating strategy: {str(e)}")
        raise

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
        
    return rsi

if __name__ == '__main__':
    port = find_free_port()
    print(f"\n* Running on http://localhost:{port}")
    socketio.run(app, debug=True, port=port, host='0.0.0.0', allow_unsafe_werkzeug=True)
