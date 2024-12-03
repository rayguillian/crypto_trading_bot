import asyncio
import websockets
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from binance import AsyncClient, BinanceSocketManager
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from strategies.base_strategy import BaseStrategy

class RealTimeTrader:
    """Real-time trading module with WebSocket support."""
    
    def __init__(self, 
                 client: AsyncClient,
                 strategy: BaseStrategy,
                 symbol: str,
                 timeframe: str = '1h',
                 trade_size: float = 100.0):
        """
        Initialize real-time trader.
        
        Args:
            client: Binance async client
            strategy: Trading strategy instance
            symbol: Trading pair symbol
            timeframe: Trading timeframe
            trade_size: Base trade size in quote currency
        """
        self.client = client
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.trade_size = trade_size
        
        self.bsm = BinanceSocketManager(self.client)
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager()
        
        self.klines_buffer = []
        self.current_position = None
        self.is_running = False
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    async def start(self):
        """Start real-time trading."""
        self.is_running = True
        
        # Start WebSocket connections
        kline_socket = self.bsm.kline_socket(
            symbol=self.symbol,
            interval=self.timeframe
        )
        
        trade_socket = self.bsm.trade_socket(self.symbol)
        
        # Create tasks for different streams
        tasks = [
            self.handle_kline_stream(kline_socket),
            self.handle_trade_stream(trade_socket),
            self.process_trading_logic()
        ]
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        
    async def handle_kline_stream(self, socket):
        """Handle incoming kline data."""
        async with socket as stream:
            while self.is_running:
                try:
                    msg = await stream.recv()
                    kline = msg['k']
                    
                    # Add to klines buffer
                    self.klines_buffer.append({
                        'timestamp': pd.Timestamp(kline['t']),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v'])
                    })
                    
                    # Keep only recent data
                    if len(self.klines_buffer) > 1000:
                        self.klines_buffer = self.klines_buffer[-1000:]
                        
                    self.logger.info(f"Received kline: {kline['c']}")
                    
                except Exception as e:
                    self.logger.error(f"Error in kline stream: {str(e)}")
                    
    async def handle_trade_stream(self, socket):
        """Handle incoming trade data."""
        async with socket as stream:
            while self.is_running:
                try:
                    msg = await stream.recv()
                    
                    # Update latest price
                    self.latest_price = float(msg['p'])
                    self.latest_volume = float(msg['q'])
                    
                    # Check for position updates
                    if self.current_position:
                        await self.check_position_update()
                        
                except Exception as e:
                    self.logger.error(f"Error in trade stream: {str(e)}")
                    
    async def process_trading_logic(self):
        """Process trading logic at regular intervals."""
        while self.is_running:
            try:
                # Convert buffer to DataFrame
                if len(self.klines_buffer) > 0:
                    df = pd.DataFrame(self.klines_buffer)
                    df.set_index('timestamp', inplace=True)
                    
                    # Generate signals
                    signals = self.strategy.generate_signals(df)
                    
                    if not signals.empty:
                        latest_signal = signals.iloc[-1]
                        
                        # Process signals
                        await self.process_signals(latest_signal)
                        
            except Exception as e:
                self.logger.error(f"Error in trading logic: {str(e)}")
                
            # Wait before next iteration
            await asyncio.sleep(1)
            
    async def process_signals(self, signal):
        """Process trading signals."""
        try:
            # Check for entry signals
            if signal['entry'] != 0 and not self.current_position:
                # Calculate position size
                risk_metrics = self.risk_manager.calculate_risk_metrics(
                    pd.DataFrame(self.klines_buffer),
                    self.latest_price
                )
                
                position_size = self.risk_manager.calculate_position_size(
                    self.trade_size,
                    self.latest_price,
                    risk_metrics
                )
                
                if position_size > 0:
                    # Place order
                    side = 'BUY' if signal['entry'] > 0 else 'SELL'
                    order = await self.client.create_order(
                        symbol=self.symbol,
                        side=side,
                        type='MARKET',
                        quantity=position_size
                    )
                    
                    self.logger.info(f"Opened {side} position: {order}")
                    
                    # Update position
                    self.current_position = {
                        'side': side,
                        'size': position_size,
                        'entry_price': float(order['fills'][0]['price'])
                    }
                    
            # Check for exit signals
            elif signal['exit'] == 1 and self.current_position:
                # Close position
                side = 'SELL' if self.current_position['side'] == 'BUY' else 'BUY'
                order = await self.client.create_order(
                    symbol=self.symbol,
                    side=side,
                    type='MARKET',
                    quantity=self.current_position['size']
                )
                
                self.logger.info(f"Closed position: {order}")
                self.current_position = None
                
        except Exception as e:
            self.logger.error(f"Error processing signals: {str(e)}")
            
    async def check_position_update(self):
        """Check and update current position."""
        if not self.current_position:
            return
            
        # Calculate current PnL
        if self.current_position['side'] == 'BUY':
            pnl = (self.latest_price - self.current_position['entry_price']) / self.current_position['entry_price']
        else:
            pnl = (self.current_position['entry_price'] - self.latest_price) / self.current_position['entry_price']
            
        # Check stop loss
        stop_loss = self.risk_manager.calculate_stop_loss(
            self.current_position['entry_price'],
            self.current_position['side'] == 'BUY'
        )
        
        if (self.current_position['side'] == 'BUY' and self.latest_price <= stop_loss) or \
           (self.current_position['side'] == 'SELL' and self.latest_price >= stop_loss):
            # Close position
            try:
                side = 'SELL' if self.current_position['side'] == 'BUY' else 'BUY'
                order = await self.client.create_order(
                    symbol=self.symbol,
                    side=side,
                    type='MARKET',
                    quantity=self.current_position['size']
                )
                
                self.logger.info(f"Stop loss triggered. Closed position: {order}")
                self.current_position = None
                
            except Exception as e:
                self.logger.error(f"Error closing position: {str(e)}")
                
    async def stop(self):
        """Stop real-time trading."""
        self.is_running = False
        await self.client.close_connection()
