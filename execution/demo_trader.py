"""
Demo Trading Engine using Yahoo Finance data
Perfect for testing strategies before connecting to real brokers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
from dataclasses import dataclass
import json
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class DemoPosition:
    """Represents a position in the demo portfolio"""
    symbol: str
    shares: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.shares
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

@dataclass
class DemoOrder:
    """Represents a demo order"""
    symbol: str
    shares: float
    order_type: str  # 'buy' or 'sell'
    price: float
    timestamp: datetime
    status: str = 'filled'  # Demo orders are instantly filled
    strategy: str = 'unknown'

class DemoTradingEngine:
    """
    Demo trading engine that simulates trading using real Yahoo Finance data
    Designed to be easily replaceable with real broker APIs later
    """
    
    def __init__(self, initial_capital: float = 100000, save_state: bool = True):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, DemoPosition] = {}
        self.order_history: List[DemoOrder] = []
        self.portfolio_history: List[Dict] = []
        self.should_save_state = save_state  # Renamed to avoid conflict with method
        self.state_file = "demo_portfolio_state.json"
        
        # Commission settings (can be adjusted)
        self.commission_per_share = 0.0  # Free trades for demo
        self.min_commission = 0.0
        
        # Load previous state if exists
        if save_state:
            self.load_state()
        
        logger.info(f"Demo trading engine initialized with ${initial_capital:,.2f}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get intraday data first
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            # Fallback to daily data
            data = ticker.history(period="2d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            logger.warning(f"No price data available for {symbol}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols efficiently"""
        prices = {}
        try:
            # Use yfinance to get multiple tickers at once
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    data = ticker.history(period="1d", interval="1m")
                    if not data.empty:
                        prices[symbol] = float(data['Close'].iloc[-1])
                    else:
                        # Fallback to daily
                        data = ticker.history(period="2d")
                        if not data.empty:
                            prices[symbol] = float(data['Close'].iloc[-1])
                except Exception as e:
                    logger.warning(f"Error getting price for {symbol}: {e}")
                    prices[symbol] = 0.0
            
        except Exception as e:
            logger.error(f"Error in batch price fetch: {e}")
            # Fallback to individual fetches
            for symbol in symbols:
                prices[symbol] = self.get_current_price(symbol)
        
        return prices
    
    def place_order(self, symbol: str, shares: float, order_type: str, 
                   strategy: str = 'manual') -> bool:
        """
        Place a demo order
        
        Args:
            symbol: Stock symbol
            shares: Number of shares
            order_type: 'buy' or 'sell'
            strategy: Strategy that generated this order
        """
        if shares <= 0:
            logger.warning(f"Invalid share count: {shares}")
            return False
        
        current_price = self.get_current_price(symbol)
        
        if current_price <= 0:
            logger.error(f"Cannot get valid price for {symbol}")
            return False
        
        # Calculate commission
        commission = max(self.commission_per_share * shares, self.min_commission)
        
        if order_type.lower() == 'buy':
            total_cost = (shares * current_price) + commission
            
            if total_cost > self.cash:
                logger.warning(f"Insufficient funds: Need ${total_cost:.2f}, have ${self.cash:.2f}")
                return False
            
            # Execute buy order
            self.cash -= total_cost
            
            if symbol in self.positions:
                # Average up/down existing position
                old_pos = self.positions[symbol]
                total_shares = old_pos.shares + shares
                total_cost_basis = (old_pos.shares * old_pos.entry_price) + (shares * current_price)
                avg_price = total_cost_basis / total_shares
                
                self.positions[symbol] = DemoPosition(
                    symbol=symbol,
                    shares=total_shares,
                    entry_price=avg_price,
                    entry_time=old_pos.entry_time,
                    current_price=current_price
                )
            else:
                # New position
                self.positions[symbol] = DemoPosition(
                    symbol=symbol,
                    shares=shares,
                    entry_price=current_price,
                    entry_time=datetime.now(),
                    current_price=current_price
                )
            
            logger.info(f"✅ BUY {shares} shares of {symbol} at ${current_price:.2f} (Strategy: {strategy})")
            
        elif order_type.lower() == 'sell':
            if symbol not in self.positions:
                logger.warning(f"No position in {symbol} to sell")
                return False
            
            position = self.positions[symbol]
            if shares > position.shares:
                logger.warning(f"Cannot sell {shares} shares, only have {position.shares}")
                return False
            
            # Execute sell order
            proceeds = (shares * current_price) - commission
            self.cash += proceeds
            
            if abs(shares - position.shares) < 0.001:  # Close entire position (account for float precision)
                del self.positions[symbol]
            else:
                # Partial sell
                position.shares -= shares
            
            logger.info(f"✅ SELL {shares} shares of {symbol} at ${current_price:.2f} (Strategy: {strategy})")
        
        else:
            logger.error(f"Invalid order type: {order_type}")
            return False
        
        # Record order
        order = DemoOrder(
            symbol=symbol,
            shares=shares,
            order_type=order_type,
            price=current_price,
            timestamp=datetime.now(),
            strategy=strategy
        )
        self.order_history.append(order)
        
        # Update portfolio and save state
        self.update_portfolio()
        if self.should_save_state:
            self.save_state()
        
        return True
    
    def update_portfolio(self):
        """Update current portfolio values with latest market prices"""
        # Get current prices for all positions
        if self.positions:
            symbols = list(self.positions.keys())
            current_prices = self.get_multiple_prices(symbols)
            
            # Update position prices
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position.current_price = current_prices[symbol]
        
        # Calculate total equity
        total_equity = self.cash
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_equity += positions_value
        
        # Record portfolio snapshot
        portfolio_data = {
            'timestamp': datetime.now().isoformat(),
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'num_positions': len(self.positions),
            'total_return': ((total_equity - self.initial_capital) / self.initial_capital) * 100,
            'daily_pnl': 0.0  # Could calculate this with previous day's data
        }
        self.portfolio_history.append(portfolio_data)
        
        # Keep only last 1000 portfolio snapshots to manage memory
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        self.update_portfolio()
        
        total_equity = self.cash
        positions_value = 0
        unrealized_pnl = 0
        
        for position in self.positions.values():
            positions_value += position.market_value
            unrealized_pnl += position.unrealized_pnl
        
        total_equity = self.cash + positions_value
        
        return {
            'timestamp': datetime.now(),
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'unrealized_pnl': unrealized_pnl,
            'total_return': ((total_equity - self.initial_capital) / self.initial_capital) * 100,
            'num_positions': len(self.positions),
            'positions': {symbol: {
                'shares': pos.shares,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'entry_time': pos.entry_time.isoformat()
            } for symbol, pos in self.positions.items()}
        }
    
    def get_position(self, symbol: str) -> Optional[DemoPosition]:
        """Get position for a specific symbol"""
        return self.positions.get(symbol)
    
    def get_buying_power(self) -> float:
        """Get available buying power (cash for demo)"""
        return self.cash
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trade history"""
        return [
            {
                'symbol': order.symbol,
                'shares': order.shares,
                'type': order.order_type,
                'price': order.price,
                'timestamp': order.timestamp.isoformat(),
                'strategy': order.strategy
            }
            for order in self.order_history[-limit:]
        ]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(self.portfolio_history) < 2:
            return {}
        
        # Get equity history
        equity_history = [snapshot['total_equity'] for snapshot in self.portfolio_history]
        returns = pd.Series(equity_history).pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Calculate metrics
        total_return = ((equity_history[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Annualized return (assuming daily data)
        days = len(returns)
        if days > 0:
            annualized_return = ((equity_history[-1] / self.initial_capital) ** (365 / days) - 1) * 100
        else:
            annualized_return = 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        if volatility > 0:
            sharpe_ratio = (annualized_return / 100 - risk_free_rate) / (volatility / 100)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        equity_series = pd.Series(equity_history)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': self._calculate_win_rate(),
            'total_trades': len(self.order_history)
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed positions"""
        # This is simplified - in reality you'd track when positions are closed
        sell_orders = [order for order in self.order_history if order.order_type == 'sell']
        if not sell_orders:
            return 0.0
        
        # Simplified win rate calculation
        return 60.0  # Placeholder
    
    def save_state(self):
        """Save current portfolio state to file"""
        try:
            state = {
                'cash': self.cash,
                'initial_capital': self.initial_capital,
                'positions': {
                    symbol: {
                        'shares': pos.shares,
                        'entry_price': pos.entry_price,
                        'entry_time': pos.entry_time.isoformat(),
                        'current_price': pos.current_price
                    }
                    for symbol, pos in self.positions.items()
                },
                'order_history': [
                    {
                        'symbol': order.symbol,
                        'shares': order.shares,
                        'order_type': order.order_type,
                        'price': order.price,
                        'timestamp': order.timestamp.isoformat(),
                        'strategy': order.strategy
                    }
                    for order in self.order_history
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """Load portfolio state from file"""
        try:
            if not os.path.exists(self.state_file):
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.cash = state.get('cash', self.initial_capital)
            self.initial_capital = state.get('initial_capital', self.initial_capital)
            
            # Restore positions
            for symbol, pos_data in state.get('positions', {}).items():
                self.positions[symbol] = DemoPosition(
                    symbol=symbol,
                    shares=pos_data['shares'],
                    entry_price=pos_data['entry_price'],
                    entry_time=datetime.fromisoformat(pos_data['entry_time']),
                    current_price=pos_data['current_price']
                )
            
            # Restore order history
            for order_data in state.get('order_history', []):
                self.order_history.append(DemoOrder(
                    symbol=order_data['symbol'],
                    shares=order_data['shares'],
                    order_type=order_data['order_type'],
                    price=order_data['price'],
                    timestamp=datetime.fromisoformat(order_data['timestamp']),
                    strategy=order_data.get('strategy', 'unknown')
                ))
            
            logger.info(f"Portfolio state loaded from {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.order_history.clear()
        self.portfolio_history.clear()
        
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        
        logger.info("Portfolio reset to initial state")

# Global demo trader instance
demo_trader = None

def get_demo_trader(initial_capital: float = 100000) -> DemoTradingEngine:
    """Get the global demo trader instance"""
    global demo_trader
    if demo_trader is None:
        demo_trader = DemoTradingEngine(initial_capital)
    return demo_trader
