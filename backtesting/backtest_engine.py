"""
Comprehensive backtesting framework for trading strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import CONFIG

@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    side: str = 'long'  # 'long' or 'short'
    strategy: str = ''
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class PortfolioSnapshot:
    date: datetime
    total_value: float
    cash: float
    positions_value: float
    positions: Dict[str, float] = field(default_factory=dict)
    drawdown: float = 0.0
    daily_return: float = 0.0

class BacktestEngine:
    """
    Comprehensive backtesting engine
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001, 
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.portfolio = self.positions  # alias for compatibility
        self.trades = []
        self.portfolio_history = []
        
        # Performance tracking
        self.daily_returns = []
        self.max_portfolio_value = initial_capital
        self.max_drawdown = 0.0
        
        # Risk management
        self.max_position_size = CONFIG.MAX_POSITION_SIZE
        self.max_daily_loss = CONFIG.MAX_DAILY_LOSS
        self.stop_loss_pct = CONFIG.STOP_LOSS_PCT
        
    def add_data(self, data: Dict[str, pd.DataFrame]):
        """Add price data for backtesting"""
        self.data = data
        self.dates = sorted(set().union(*[df.index for df in data.values()]))
    
    def run_backtest(self, strategy, start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest for a strategy
        """
        print(f"Starting backtest for {strategy.name}")
        
        # Filter dates
        if start_date:
            start_date = pd.to_datetime(start_date)
            self.dates = [d for d in self.dates if d >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            self.dates = [d for d in self.dates if d <= end_date]
        
        # Initialize portfolio
        self._reset_portfolio()
        
        # Run backtest day by day
        for i, current_date in enumerate(self.dates):
            try:
                # Get current market data
                current_data = self._get_market_data_at_date(current_date)
                
                if not current_data:
                    continue
                
                # Update portfolio value
                portfolio_value = self._calculate_portfolio_value(current_data)
                
                # Check risk limits
                if self._check_risk_limits(portfolio_value):
                    continue
                
                # Generate signals
                signals = strategy.generate_signals(self._get_historical_data_up_to_date(current_date))
                
                # Execute trades
                for signal in signals:
                    self._execute_signal(signal, current_data, portfolio_value)
                
                # Update stop losses and take profits
                self._update_exit_orders(current_data)
                
                # Record portfolio snapshot
                self._record_portfolio_snapshot(current_date, current_data)
                
                # Print progress
                if i % 50 == 0:
                    print(f"Progress: {i+1}/{len(self.dates)} ({(i+1)/len(self.dates)*100:.1f}%)")
                    
            except Exception as e:
                print(f"Error on {current_date}: {e}")
                continue
        
        print(f"Backtest completed. Total trades: {len(self.trades)}")
        return self._calculate_performance_metrics()
    
    def _reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
    
    def _get_market_data_at_date(self, date: datetime) -> Dict:
        """Get market data for all symbols at specific date"""
        market_data = {}
        for symbol, df in self.data.items():
            if date in df.index:
                market_data[symbol] = df.loc[date]
        return market_data
    
    def _get_historical_data_up_to_date(self, date: datetime) -> Dict:
        """Get historical data up to specific date"""
        historical_data = {}
        for symbol, df in self.data.items():
            historical_data[symbol] = df[df.index <= date]
        return historical_data
    
    def _calculate_portfolio_value(self, current_data: Dict) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        for symbol, quantity in self.positions.items():
            if symbol in current_data:
                positions_value += quantity * current_data[symbol]['Close']
        
        return self.cash + positions_value
    
    def _check_risk_limits(self, portfolio_value: float) -> bool:
        """Check if risk limits are breached"""
        # Check daily loss limit
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1].total_value
            daily_loss = (prev_value - portfolio_value) / prev_value
            
            if daily_loss > self.max_daily_loss:
                print(f"Daily loss limit breached: {daily_loss:.2%}")
                # Close all positions
                self._close_all_positions()
                return True
        
        return False
    
    def _execute_signal(self, signal, current_data: Dict, portfolio_value: float):
        """Execute a trading signal"""
        if signal.symbol not in current_data:
            return
        
        # Calculate position size
        position_size = signal.metadata.get('position_size')
        if not position_size:
            # Use strategy's position sizing if not provided
            volatility = self._calculate_volatility(signal.symbol)
            position_size = signal.strength * 0.1 * portfolio_value  # Default 10% max
        
        # Check position size limits
        max_position_value = portfolio_value * self.max_position_size
        position_size = min(position_size, max_position_value)
        
        current_price = current_data[signal.symbol]['Close']
        quantity = position_size / current_price
        
        # Calculate costs
        commission_cost = position_size * self.commission
        slippage_cost = position_size * self.slippage
        total_cost = commission_cost + slippage_cost
        
        # Check if we have enough cash
        if signal.signal_type.value > 0:  # Buy signal
            if self.cash < position_size + total_cost:
                return  # Not enough cash
            
            # Execute buy
            self.cash -= (position_size + total_cost)
            self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + quantity
            
        else:  # Sell signal
            current_position = self.positions.get(signal.symbol, 0)
            if current_position <= 0:
                return  # No position to sell
            
            # Execute sell
            sell_quantity = min(quantity, current_position)
            sell_value = sell_quantity * current_price
            
            self.cash += (sell_value - total_cost)
            self.positions[signal.symbol] -= sell_quantity
            
            if abs(self.positions[signal.symbol]) < 1e-6:  # Close to zero
                del self.positions[signal.symbol]
        
        # Record trade
        trade = Trade(
            symbol=signal.symbol,
            entry_date=signal.timestamp,
            entry_price=current_price,
            quantity=quantity if signal.signal_type.value > 0 else -quantity,
            side='long' if signal.signal_type.value > 0 else 'short',
            strategy=signal.metadata.get('strategy', 'unknown'),
            commission=commission_cost,
            slippage=slippage_cost,
            metadata=signal.metadata
        )
        
        self.trades.append(trade)
    
    def _update_exit_orders(self, current_data: Dict):
        """Update stop losses and take profits"""
        for symbol, quantity in list(self.positions.items()):
            if symbol not in current_data or quantity == 0:
                continue
            
            current_price = current_data[symbol]['Close']
            
            # Find the most recent entry for this symbol
            recent_trades = [t for t in self.trades if t.symbol == symbol and t.exit_date is None]
            
            if not recent_trades:
                continue
            
            latest_trade = recent_trades[-1]
            
            # Check stop loss
            if quantity > 0:  # Long position
                stop_loss_price = latest_trade.entry_price * (1 - self.stop_loss_pct)
                if current_price <= stop_loss_price:
                    self._close_position(symbol, current_price, 'stop_loss')
            else:  # Short position
                stop_loss_price = latest_trade.entry_price * (1 + self.stop_loss_pct)
                if current_price >= stop_loss_price:
                    self._close_position(symbol, current_price, 'stop_loss')
    
    def _close_position(self, symbol: str, exit_price: float, exit_reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return
        
        quantity = self.positions[symbol]
        position_value = abs(quantity) * exit_price
        
        # Calculate costs
        commission_cost = position_value * self.commission
        slippage_cost = position_value * self.slippage
        total_cost = commission_cost + slippage_cost
        
        # Update cash
        if quantity > 0:  # Closing long position
            self.cash += (position_value - total_cost)
        else:  # Closing short position
            self.cash -= (position_value + total_cost)
        
        # Remove position
        del self.positions[symbol]
        
        # Update trade records
        for trade in reversed(self.trades):
            if trade.symbol == symbol and trade.exit_date is None:
                trade.exit_date = datetime.now()
                trade.exit_price = exit_price
                trade.pnl = self._calculate_trade_pnl(trade)
                trade.metadata['exit_reason'] = exit_reason
                break
    
    def _close_all_positions(self):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            if symbol in self.data:
                latest_price = self.data[symbol]['Close'].iloc[-1]
                self._close_position(symbol, latest_price, 'risk_limit')
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade"""
        if trade.exit_price is None:
            return 0.0
        
        if trade.side == 'long':
            pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        else:  # short
            pnl = (trade.entry_price - trade.exit_price) * abs(trade.quantity)
        
        return pnl - trade.commission - trade.slippage
    
    def _calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """Calculate volatility for position sizing"""
        if symbol not in self.data:
            return 0.2  # Default volatility
        
        returns = self.data[symbol]['Close'].pct_change().dropna()
        if len(returns) < window:
            return 0.2
        
        return returns.tail(window).std() * np.sqrt(252)
    
    def _record_portfolio_snapshot(self, date: datetime, current_data: Dict):
        """Record portfolio snapshot"""
        portfolio_value = self._calculate_portfolio_value(current_data)
        
        # Calculate daily return
        daily_return = 0.0
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1].total_value
            daily_return = (portfolio_value - prev_value) / prev_value
        
        # Update max drawdown
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Record snapshot
        snapshot = PortfolioSnapshot(
            date=date,
            total_value=portfolio_value,
            cash=self.cash,
            positions_value=portfolio_value - self.cash,
            positions=self.positions.copy(),
            drawdown=drawdown,
            daily_return=daily_return
        )
        
        self.portfolio_history.append(snapshot)
        self.daily_returns.append(daily_return)
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_history:
            return {}
        
        # Convert to series
        portfolio_values = pd.Series([p.total_value for p in self.portfolio_history])
        dates = pd.Series([p.date for p in self.portfolio_history])
        daily_returns = pd.Series(self.daily_returns[1:])  # Skip first day
        
        # Calculate metrics
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Win rate
        winning_trades = [t for t in self.trades if self._calculate_trade_pnl(t) > 0]
        total_completed_trades = [t for t in self.trades if t.exit_date is not None]
        win_rate = len(winning_trades) / len(total_completed_trades) if total_completed_trades else 0
        
        # Average trade metrics
        trade_pnls = [self._calculate_trade_pnl(t) for t in total_completed_trades]
        avg_trade_return = np.mean(trade_pnls) if trade_pnls else 0
        avg_winning_trade = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if any(pnl > 0 for pnl in trade_pnls) else 0
        avg_losing_trade = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if any(pnl < 0 for pnl in trade_pnls) else 0
        
        profit_factor = abs(avg_winning_trade * len(winning_trades) / (avg_losing_trade * (len(total_completed_trades) - len(winning_trades)))) if avg_losing_trade != 0 else float('inf')
        
        # Drawdown metrics
        drawdowns = [p.drawdown for p in self.portfolio_history]
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': portfolio_values.iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(total_completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(total_completed_trades) - len(winning_trades),
            'avg_trade_return': avg_trade_return,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history
        }
    
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """Generate a comprehensive backtest report"""
        report = []
        report.append("=" * 60)
        report.append("ALGORITHMIC TRADING BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Initial Capital: ${results['initial_capital']:,.2f}")
        report.append(f"Final Value: ${results['final_value']:,.2f}")
        report.append(f"Total Return: {results['total_return']:.2%}")
        report.append(f"Annualized Return: {results['annualized_return']:.2%}")
        report.append(f"Volatility: {results['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        report.append(f"Sortino Ratio: {results['sortino_ratio']:.3f}")
        report.append(f"Calmar Ratio: {results['calmar_ratio']:.3f}")
        report.append(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        report.append("")
        
        # Trading Statistics
        report.append("TRADING STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Trades: {results['total_trades']}")
        report.append(f"Winning Trades: {results['winning_trades']}")
        report.append(f"Losing Trades: {results['losing_trades']}")
        report.append(f"Win Rate: {results['win_rate']:.2%}")
        report.append(f"Average Trade Return: ${results['avg_trade_return']:.2f}")
        report.append(f"Average Winning Trade: ${results['avg_winning_trade']:.2f}")
        report.append(f"Average Losing Trade: ${results['avg_losing_trade']:.2f}")
        report.append(f"Profit Factor: {results['profit_factor']:.3f}")
        report.append("")
        
        # Strategy Breakdown
        strategy_stats = self._get_strategy_breakdown(results['trades'])
        if strategy_stats:
            report.append("STRATEGY BREAKDOWN")
            report.append("-" * 30)
            for strategy, stats in strategy_stats.items():
                report.append(f"{strategy}: {stats['trades']} trades, "
                            f"{stats['win_rate']:.1%} win rate, "
                            f"${stats['avg_pnl']:.2f} avg P&L")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def _get_strategy_breakdown(self, trades: List[Trade]) -> Dict:
        """Get performance breakdown by strategy"""
        strategy_stats = {}
        
        for trade in trades:
            if trade.exit_date is None:
                continue
                
            strategy = trade.strategy
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0
                }
            
            pnl = self._calculate_trade_pnl(trade)
            strategy_stats[strategy]['trades'] += 1
            strategy_stats[strategy]['total_pnl'] += pnl
            if pnl > 0:
                strategy_stats[strategy]['wins'] += 1
        
        # Calculate derived metrics
        for strategy, stats in strategy_stats.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
        
        return strategy_stats
