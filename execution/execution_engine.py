"""
Live Trading Execution Engine
Handles order execution and portfolio management for live trading
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
from config.config import CONFIG

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: Alpaca API not available. Paper trading only.")

@dataclass
class Order:
    symbol: str
    quantity: float
    side: str  # 'buy' or 'sell'
    order_type: str = 'market'  # 'market', 'limit', 'stop', 'stop_limit'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'day'  # 'day', 'gtc', 'ioc', 'fok'
    strategy: str = ''
    metadata: Dict = None

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    side: str

class RiskManager:
    """
    Risk management for live trading
    """
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.daily_pnl = 0.0
        self.position_sizes = {}
        self.correlation_matrix = pd.DataFrame()
        
    def check_order_risk(self, order: Order, portfolio_value: float, 
                        current_positions: Dict[str, Position]) -> bool:
        """
        Check if order passes risk management criteria
        """
        # Check position size limit
        order_value = abs(order.quantity * (order.limit_price or 100))  # Estimate
        position_pct = order_value / portfolio_value
        
        if position_pct > self.config.MAX_POSITION_SIZE:
            logging.warning(f"Order rejected: Position size {position_pct:.2%} exceeds limit {self.config.MAX_POSITION_SIZE:.2%}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.MAX_DAILY_LOSS * portfolio_value:
            logging.warning(f"Order rejected: Daily loss limit exceeded")
            return False
        
        # Check correlation with existing positions
        if self._check_correlation_risk(order.symbol, current_positions):
            logging.warning(f"Order rejected: Correlation risk too high for {order.symbol}")
            return False
        
        return True
    
    def _check_correlation_risk(self, symbol: str, positions: Dict[str, Position]) -> bool:
        """
        Check if adding position would create excessive correlation risk
        """
        if len(positions) < 2:
            return False
            
        # This would need historical correlation data
        # Simplified version - reject if same sector concentration > threshold
        # In practice, you'd use correlation matrix
        return False
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L"""
        self.daily_pnl += pnl_change
    
    def reset_daily_pnl(self):
        """Reset daily P&L at market open"""
        self.daily_pnl = 0.0

class ExecutionEngine:
    """
    Main execution engine for live trading
    """
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.risk_manager = RiskManager()
        
        # Initialize broker connection
        if ALPACA_AVAILABLE and not paper_trading:
            self.api = tradeapi.REST(
                CONFIG.ALPACA_API_KEY,
                CONFIG.ALPACA_SECRET_KEY,
                CONFIG.ALPACA_BASE_URL,
                api_version='v2'
            )
        else:
            self.api = None
            print("Running in paper trading mode")
        
        # Portfolio state
        self.positions = {}
        self.pending_orders = {}
        self.executed_orders = []
        
        # Performance tracking
        self.portfolio_value_history = []
        self.trade_history = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading.log'),
                logging.StreamHandler()
            ]
        )
        
    def initialize(self):
        """Initialize the execution engine"""
        if self.api:
            try:
                # Get account info
                account = self.api.get_account()
                logging.info(f"Account initialized: ${float(account.portfolio_value):,.2f} portfolio value")
                
                # Get current positions
                positions = self.api.list_positions()
                for pos in positions:
                    self.positions[pos.symbol] = Position(
                        symbol=pos.symbol,
                        quantity=float(pos.qty),
                        avg_price=float(pos.avg_cost),
                        market_value=float(pos.market_value),
                        unrealized_pnl=float(pos.unrealized_pl),
                        side=pos.side
                    )
                
                logging.info(f"Loaded {len(self.positions)} existing positions")
                
            except Exception as e:
                logging.error(f"Failed to initialize API: {e}")
                raise
        else:
            logging.info("Paper trading mode initialized")
    
    def execute_signals(self, signals: List, current_prices: Dict[str, float]):
        """
        Execute trading signals
        """
        portfolio_value = self.get_portfolio_value()
        
        for signal in signals:
            try:
                order = self._convert_signal_to_order(signal, portfolio_value, current_prices)
                if order and self.risk_manager.check_order_risk(order, portfolio_value, self.positions):
                    self._submit_order(order)
                    
            except Exception as e:
                logging.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def _convert_signal_to_order(self, signal, portfolio_value: float, 
                               current_prices: Dict[str, float]) -> Optional[Order]:
        """
        Convert trading signal to order
        """
        if signal.symbol not in current_prices:
            logging.warning(f"No price data for {signal.symbol}")
            return None
        
        current_price = current_prices[signal.symbol]
        
        # Calculate position size
        if hasattr(signal, 'position_size') and signal.position_size:
            position_size = signal.position_size
        else:
            # Default position sizing based on signal strength
            max_position_value = portfolio_value * CONFIG.MAX_POSITION_SIZE
            position_value = max_position_value * signal.strength
            position_size = position_value / current_price
        
        # Determine order side
        if signal.signal_type.value > 0:  # Buy signal
            side = 'buy'
            quantity = abs(position_size)
        else:  # Sell signal
            side = 'sell'
            # Check if we have existing position
            current_position = self.positions.get(signal.symbol)
            if not current_position or current_position.quantity <= 0:
                logging.warning(f"Cannot sell {signal.symbol}: No long position")
                return None
            quantity = min(abs(position_size), current_position.quantity)
        
        # Create order with stop loss and take profit
        order = Order(
            symbol=signal.symbol,
            quantity=quantity,
            side=side,
            order_type='market',  # Can be enhanced to use limit orders
            strategy=signal.metadata.get('strategy', 'unknown'),
            metadata={
                'signal_strength': signal.strength,
                'stop_loss_price': self._calculate_stop_loss_price(signal, current_price),
                'take_profit_price': self._calculate_take_profit_price(signal, current_price),
                **signal.metadata
            }
        )
        
        return order
    
    def _calculate_stop_loss_price(self, signal, current_price: float) -> float:
        """Calculate stop loss price"""
        atr = signal.metadata.get('atr', current_price * 0.02)  # Default 2% if no ATR
        
        if signal.signal_type.value > 0:  # Long position
            return current_price - (atr * 2)  # 2x ATR stop loss
        else:  # Short position
            return current_price + (atr * 2)
    
    def _calculate_take_profit_price(self, signal, current_price: float) -> float:
        """Calculate take profit price"""
        atr = signal.metadata.get('atr', current_price * 0.02)
        
        if signal.signal_type.value > 0:  # Long position
            return current_price + (atr * 4)  # 4x ATR take profit (2:1 risk-reward)
        else:  # Short position
            return current_price - (atr * 4)
    
    def _submit_order(self, order: Order):
        """
        Submit order to broker or paper trading system
        """
        if self.api:
            try:
                # Submit market order
                submitted_order = self.api.submit_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=order.side,
                    type=order.order_type,
                    time_in_force=order.time_in_force
                )
                
                logging.info(f"Order submitted: {order.side} {order.quantity} {order.symbol} at market")
                
                # Store pending order
                self.pending_orders[submitted_order.id] = {
                    'order': order,
                    'alpaca_order': submitted_order,
                    'timestamp': datetime.now()
                }
                
                # Submit stop loss and take profit orders after main order fills
                self._schedule_bracket_orders(order, submitted_order)
                
            except Exception as e:
                logging.error(f"Failed to submit order: {e}")
        else:
            # Paper trading simulation
            self._simulate_order_execution(order)
    
    def _schedule_bracket_orders(self, original_order: Order, parent_order):
        """
        Schedule stop loss and take profit orders
        """
        # This would be enhanced to monitor order fills and submit bracket orders
        # For now, we'll simulate immediate execution
        pass
    
    def _simulate_order_execution(self, order: Order):
        """
        Simulate order execution for paper trading
        """
        # Simulate execution with slight slippage
        slippage = 0.001  # 0.1% slippage
        execution_price = order.limit_price or 100  # Would need current market price
        
        if order.side == 'buy':
            execution_price *= (1 + slippage)
        else:
            execution_price *= (1 - slippage)
        
        # Update positions
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            if order.side == 'buy':
                new_qty = position.quantity + order.quantity
                new_avg_price = ((position.quantity * position.avg_price) + 
                               (order.quantity * execution_price)) / new_qty
                position.quantity = new_qty
                position.avg_price = new_avg_price
            else:  # sell
                position.quantity -= order.quantity
                if position.quantity <= 0:
                    del self.positions[order.symbol]
        else:
            if order.side == 'buy':
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=execution_price,
                    market_value=order.quantity * execution_price,
                    unrealized_pnl=0.0,
                    side='long'
                )
        
        # Record execution
        self.executed_orders.append({
            'order': order,
            'execution_price': execution_price,
            'timestamp': datetime.now()
        })
        
        logging.info(f"Paper trade executed: {order.side} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update position values with current market prices
        """
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value
        """
        if self.api:
            try:
                account = self.api.get_account()
                return float(account.portfolio_value)
            except:
                pass
        
        # Paper trading calculation
        total_value = 0.0
        for position in self.positions.values():
            total_value += position.market_value
        
        return total_value or CONFIG.INITIAL_CAPITAL
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary
        """
        portfolio_value = self.get_portfolio_value()
        
        summary = {
            'total_value': portfolio_value,
            'positions': len(self.positions),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'daily_pnl': self.risk_manager.daily_pnl,
            'position_details': {}
        }
        
        for symbol, position in self.positions.items():
            summary['position_details'][symbol] = {
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'weight': position.market_value / portfolio_value if portfolio_value > 0 else 0
            }
        
        return summary
    
    def check_and_update_stops(self, current_prices: Dict[str, float]):
        """
        Check and update stop loss orders
        """
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Simple stop loss check (would be enhanced with trailing stops)
            if position.quantity > 0:  # Long position
                stop_loss_pct = CONFIG.STOP_LOSS_PCT
                stop_price = position.avg_price * (1 - stop_loss_pct)
                
                if current_price <= stop_price:
                    self._submit_stop_loss_order(symbol, position.quantity)
            
            # Similar logic for short positions...
    
    def _submit_stop_loss_order(self, symbol: str, quantity: float):
        """
        Submit stop loss order
        """
        stop_order = Order(
            symbol=symbol,
            quantity=quantity,
            side='sell',
            order_type='market',
            strategy='stop_loss'
        )
        
        self._submit_order(stop_order)
        logging.warning(f"Stop loss triggered for {symbol}")
    
    def shutdown(self):
        """
        Shutdown execution engine
        """
        # Close all positions if needed
        # Cancel pending orders
        # Save state
        logging.info("Execution engine shutdown")

# Global execution engine instance
execution_engine = ExecutionEngine(paper_trading=True)
