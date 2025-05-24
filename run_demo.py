#!/usr/bin/env python3
"""
Demo Trading Bot Launcher - Uses Yahoo Finance data
Perfect for testing strategies before connecting to real brokers like Interactive Brokers
"""

import os
import sys
import time
import signal
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from execution.demo_trader import DemoTradingEngine
from data.data_manager import DataManager
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.ml_ensemble import MLEnsembleStrategy
from config.config import MEAN_REVERSION_CONFIG, MOMENTUM_CONFIG, ML_CONFIG
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DemoTradingBot:
    """
    Demo trading bot that uses Yahoo Finance data
    Can be easily switched to Interactive Brokers later
    """
    
    def __init__(self, initial_capital: float = 500):
        self.demo_trader = DemoTradingEngine(initial_capital)
        self.data_manager = DataManager()
        self.running = False
        
        # Initialize strategies
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(MEAN_REVERSION_CONFIG),
            'momentum': MomentumStrategy(MOMENTUM_CONFIG),
            'ml_ensemble': MLEnsembleStrategy(ML_CONFIG)
        }
        
        # Trading symbols (smaller, more affordable stocks for $500 capital)
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL',                      # High-priced but fractional shares
            'AMD', 'NVDA', 'TSM',                         # Tech stocks
            'SHOP.TO', 'TD.TO',                           # Canadian stocks
            'SPY', 'QQQ', 'VTI',                          # ETFs
            'F', 'GE', 'T'                                # Lower-priced stocks
        ]
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Strategy allocation (what % of capital to allocate to each strategy)
        self.strategy_allocation = {
            'mean_reversion': 0.4,  # 40%
            'momentum': 0.4,        # 40%
            'ml_ensemble': 0.2      # 20%
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received shutdown signal ({signum}), stopping bot...")
        self.running = False
    
    def display_banner(self):
        """Display welcome banner"""
        print("\n" + "="*70)
        print("🤖 ALGORITHMIC TRADING BOT - DEMO MODE")
        print("📊 Using Yahoo Finance Real-Time Data")
        print("🇨🇦 Ready for Interactive Brokers Integration Later")
        print("="*70)
        print(f"💰 Initial Capital: ${self.demo_trader.initial_capital:,.2f}")
        print(f"📈 Trading Universe: {len(self.symbols)} symbols")
        print(f"🧠 Active Strategies: {', '.join(self.strategies.keys())}")
        print(f"🕐 Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    
    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch current market data for all symbols"""
        print("📊 Fetching market data...")
        market_data = {}
        
        for symbol in self.symbols:
            try:
                # Get recent historical data for strategy analysis
                data = self.data_manager.get_historical_data(
                    [symbol], 
                    period='30d'  # 30 days of data
                )
                
                if symbol in data and not data[symbol].empty:
                    market_data[symbol] = data[symbol]
                    current_price = data[symbol]['Close'].iloc[-1]
                    print(f"   ✅ {symbol}: ${current_price:.2f}")
                else:
                    print(f"   ❌ {symbol}: No data available")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {e}")
        
        print(f"📊 Successfully fetched data for {len(market_data)} symbols\n")
        return market_data
    
    def generate_trading_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate trading signals from all strategies"""
        print("🧠 Generating trading signals...")
        all_signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                print(f"   🔍 Running {strategy_name} strategy...")
                
                # Generate signals for each symbol
                for symbol, data in market_data.items():
                    if len(data) < 20:  # Need enough data for analysis
                        continue
                    
                    # Each strategy generates its own signals
                    signal = strategy.generate_signal(data)
                    
                    if signal and signal.get('action') in ['buy', 'sell']:
                        signal['symbol'] = symbol
                        signal['strategy'] = strategy_name
                        signal['timestamp'] = datetime.now()
                        signal['price'] = data['Close'].iloc[-1]
                        all_signals.append(signal)
                        
                        action = signal['action'].upper()
                        confidence = signal.get('confidence', 0.5)
                        print(f"      📍 {symbol}: {action} (confidence: {confidence:.2f})")
                
            except Exception as e:
                print(f"   ❌ Error in {strategy_name}: {e}")
        
        print(f"🧠 Generated {len(all_signals)} trading signals\n")
        return all_signals
    
    def execute_trades(self, signals: List[Dict]):
        """Execute trades based on signals"""
        if not signals:
            print("💼 No trades to execute\n")
            return
        
        print("💼 Executing trades...")
        portfolio_value = self.demo_trader.get_portfolio_summary()['total_equity']
        
        # Group signals by strategy for proper allocation
        strategy_signals = {}
        for signal in signals:
            strategy = signal['strategy']
            if strategy not in strategy_signals:
                strategy_signals[strategy] = []
            strategy_signals[strategy].append(signal)
        
        for strategy_name, strategy_signals_list in strategy_signals.items():
            # Calculate available capital for this strategy
            strategy_capital = portfolio_value * self.strategy_allocation[strategy_name]
            
            for signal in strategy_signals_list:
                symbol = signal['symbol']
                action = signal['action']
                confidence = signal.get('confidence', 0.5)
                price = signal['price']
                
                # Only execute high-confidence signals
                if confidence < 0.6:
                    continue
                
                if action == 'buy':
                    # Calculate position size (10% of strategy capital per position for small capital)
                    position_value = strategy_capital * 0.10  # Increased from 0.02 for small capital
                    shares = int(position_value / price)
                    
                    if shares > 0 and position_value <= self.demo_trader.cash:
                        success = self.demo_trader.place_order(
                            symbol, shares, 'buy', strategy_name
                        )
                        if success:
                            print(f"      ✅ Bought {shares} shares of {symbol} @ ${price:.2f}")
                
                elif action == 'sell':
                    # Check if we have a position to sell
                    position = self.demo_trader.get_position(symbol)
                    if position and position.shares > 0:
                        # Sell 50% of position
                        shares_to_sell = max(1, int(position.shares * 0.5))
                        success = self.demo_trader.place_order(
                            symbol, shares_to_sell, 'sell', strategy_name
                        )
                        if success:
                            print(f"      ✅ Sold {shares_to_sell} shares of {symbol} @ ${price:.2f}")
        
        print("💼 Trade execution completed\n")
    
    def display_portfolio_status(self):
        """Display current portfolio status"""
        summary = self.demo_trader.get_portfolio_summary()
        
        print("💼 PORTFOLIO STATUS")
        print("-" * 50)
        print(f"💵 Cash: ${summary['cash']:,.2f}")
        print(f"📈 Positions Value: ${summary['positions_value']:,.2f}")
        print(f"💰 Total Equity: ${summary['total_equity']:,.2f}")
        print(f"📊 Total Return: {summary['total_return']:+.2f}%")
        print(f"📋 Active Positions: {summary['num_positions']}")
        
        if summary['unrealized_pnl'] != 0:
            pnl_color = "📈" if summary['unrealized_pnl'] > 0 else "📉"
            print(f"{pnl_color} Unrealized P&L: ${summary['unrealized_pnl']:+,.2f}")
        
        # Show top positions
        if summary['positions']:
            print(f"\n📋 CURRENT POSITIONS:")
            sorted_positions = sorted(
                summary['positions'].items(),
                key=lambda x: x[1]['market_value'],
                reverse=True
            )
            
            for symbol, pos in sorted_positions[:5]:  # Show top 5
                pnl_pct = pos['unrealized_pnl_pct']
                pnl_indicator = "📈" if pnl_pct > 0 else "📉" if pnl_pct < 0 else "➡️"
                print(f"   {pnl_indicator} {symbol}: {pos['shares']:.0f} shares @ ${pos['current_price']:.2f} "
                      f"(P&L: {pnl_pct:+.1f}%)")
        
        print()
    
    def display_recent_activity(self):
        """Display recent trading activity"""
        recent_trades = self.demo_trader.get_recent_trades(limit=5)
        
        if recent_trades:
            print("📋 RECENT TRADES")
            print("-" * 50)
            for trade in recent_trades[-5:]:
                trade_time = datetime.fromisoformat(trade['timestamp']).strftime('%H:%M:%S')
                action_icon = "🟢" if trade['type'] == 'buy' else "🔴"
                print(f"   {action_icon} {trade_time} - {trade['type'].upper()} "
                      f"{trade['shares']:.0f} {trade['symbol']} @ ${trade['price']:.2f} "
                      f"({trade['strategy']})")
            print()
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        cycle_start = datetime.now()
        print(f"\n{'='*20} TRADING CYCLE {cycle_start.strftime('%H:%M:%S')} {'='*20}")
        
        try:
            # 1. Fetch market data
            market_data = self.fetch_market_data()
            
            if not market_data:
                print("❌ No market data available, skipping cycle\n")
                return
            
            # 2. Generate trading signals
            signals = self.generate_trading_signals(market_data)
            
            # 3. Execute trades
            self.execute_trades(signals)
            
            # 4. Update and display portfolio
            self.display_portfolio_status()
            self.display_recent_activity()
            
            # 5. Performance metrics
            metrics = self.demo_trader.get_performance_metrics()
            if metrics:
                print("📊 PERFORMANCE METRICS")
                print("-" * 50)
                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
                print(f"   Total Trades: {metrics.get('total_trades', 0)}")
                print()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            print(f"⏱️  Cycle completed in {cycle_duration:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            print(f"❌ Error in trading cycle: {e}")
    
    def run(self, cycle_interval_minutes: int = 1):
        """
        Main trading loop
        
        Args:
            cycle_interval_minutes: How often to run trading cycles (default: 1 minute)
        """
        self.display_banner()
        self.running = True
        cycle_count = 0
        
        # Display initial portfolio state
        self.display_portfolio_status()
        
        try:
            while self.running:
                cycle_count += 1
                
                # Run trading cycle
                self.run_trading_cycle()
                
                if not self.running:
                    break
                
                # Wait for next cycle
                print(f"⏰ Waiting {cycle_interval_minutes} minute(s) until next cycle...")
                
                for i in range(cycle_interval_minutes * 60):  # Convert to seconds
                    if not self.running:
                        break
                    time.sleep(1)
                    
                    # Show countdown every 30 seconds for 1-minute intervals
                    if cycle_interval_minutes == 1 and i % 30 == 0 and i > 0:
                        remaining_seconds = (cycle_interval_minutes * 60 - i)
                        if remaining_seconds > 0:
                            print(f"   ⏱️  {remaining_seconds} seconds remaining...")
                    # Show countdown every minute for longer intervals
                    elif cycle_interval_minutes > 1 and i % 60 == 0 and i > 0:
                        remaining_minutes = (cycle_interval_minutes * 60 - i) // 60
                        if remaining_minutes > 0:
                            print(f"   ⏱️  {remaining_minutes} minutes remaining...")
        
        except KeyboardInterrupt:
            print("\n🛑 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"❌ Unexpected error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the trading bot gracefully"""
        print(f"\n🔄 Shutting down demo trading bot...")
        
        # Display final summary
        final_summary = self.demo_trader.get_portfolio_summary()
        metrics = self.demo_trader.get_performance_metrics()
        
        print("\n" + "="*60)
        print("📊 FINAL SESSION SUMMARY")
        print("="*60)
        print(f"💰 Final Equity: ${final_summary['total_equity']:,.2f}")
        print(f"📈 Total Return: {final_summary['total_return']:+.2f}%")
        print(f"💵 Cash: ${final_summary['cash']:,.2f}")
        print(f"📋 Final Positions: {final_summary['num_positions']}")
        
        if metrics:
            print(f"📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"🔄 Total Trades: {metrics.get('total_trades', 0)}")
        
        # Show final positions
        if final_summary['positions']:
            print(f"\n📋 FINAL POSITIONS:")
            for symbol, pos in final_summary['positions'].items():
                pnl_pct = pos['unrealized_pnl_pct']
                pnl_indicator = "📈" if pnl_pct > 0 else "📉"
                print(f"   {pnl_indicator} {symbol}: {pos['shares']:.0f} shares "
                      f"(P&L: {pnl_pct:+.1f}%)")
        
        print("\n✅ Demo trading session completed!")
        print("🔄 Ready to connect to Interactive Brokers when you're ready!")
        print("="*60)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Trading Bot')
    parser.add_argument('--capital', type=float, default=500,
                       help='Initial capital (default: $500)')
    parser.add_argument('--interval', type=int, default=1,
                       help='Trading cycle interval in minutes (default: 1)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset portfolio to initial state')
    
    args = parser.parse_args()
    
    print("🚀 Starting Demo Trading Bot...")
    
    bot = DemoTradingBot(args.capital)
    
    if args.reset:
        bot.demo_trader.reset_portfolio()
        print("🔄 Portfolio reset to initial state")
    
    try:
        bot.run(cycle_interval_minutes=args.interval)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"💥 Fatal error: {e}")

if __name__ == "__main__":
    main()
