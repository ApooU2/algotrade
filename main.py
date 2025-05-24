"""
Main Trading Bot - Orchestrates the entire trading system
"""
import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import signal
import sys
import os

from strategies.strategy_manager import strategy_manager
from data.data_manager import data_manager
from execution.execution_engine import execution_engine
from backtesting.backtest_engine import BacktestEngine
from config.config import CONFIG

class TradingBot:
    """
    Main trading bot that coordinates all components
    """
    
    def __init__(self, mode='live'):  # 'live', 'paper', 'backtest'
        self.mode = mode
        self.running = False
        self.last_market_data_update = None
        self.market_data = {}
        
        # Setup logging
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradingBot')
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def initialize(self):
        """Initialize all components"""
        self.logger.info(f"Initializing Trading Bot in {self.mode} mode")
        
        try:
            # Initialize execution engine
            execution_engine.initialize()
            
            # Load initial market data
            self._update_market_data()
            
            # Schedule regular tasks
            self._schedule_tasks()
            
            self.logger.info("Trading Bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Trading Bot: {e}")
            raise
    
    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Run comprehensive backtesting
        """
        self.logger.info("Starting comprehensive backtest")
        
        # Use default dates if not provided
        if not start_date:
            start_date = CONFIG.BACKTEST_START
        if not end_date:
            end_date = CONFIG.BACKTEST_END
        
        # Fetch historical data
        symbols = CONFIG.SYMBOLS
        self.logger.info(f"Fetching historical data for {len(symbols)} symbols")
        
        historical_data = data_manager.fetch_historical_data(
            symbols=symbols,
            period='5y',  # 5 years of data
            interval='1d'
        )
        
        if not historical_data:
            raise ValueError("No historical data available for backtesting")
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(
            initial_capital=CONFIG.INITIAL_CAPITAL,
            commission=CONFIG.COMMISSION
        )
        
        backtest_engine.add_data(historical_data)
        
        # Test each strategy individually
        individual_results = {}
        
        for strategy_name, strategy in strategy_manager.strategies.items():
            self.logger.info(f"Backtesting {strategy_name}")
            
            # Reset backtest engine for each strategy
            backtest_engine._reset_portfolio()
            
            try:
                results = backtest_engine.run_backtest(
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date
                )
                individual_results[strategy_name] = results
                
                self.logger.info(f"{strategy_name} - Total Return: {results['total_return']:.2%}, "
                               f"Sharpe: {results['sharpe_ratio']:.3f}, "
                               f"Max DD: {results['max_drawdown']:.2%}")
                
            except Exception as e:
                self.logger.error(f"Error backtesting {strategy_name}: {e}")
                continue
        
        # Test combined strategy
        self.logger.info("Backtesting combined strategy")
        backtest_engine._reset_portfolio()
        
        # Create a combined strategy that uses strategy_manager
        class CombinedStrategy:
            def __init__(self):
                self.name = "Combined Strategy"
            
            def generate_signals(self, data):
                return strategy_manager.generate_combined_signals(data)
        
        combined_strategy = CombinedStrategy()
        combined_results = backtest_engine.run_backtest(
            strategy=combined_strategy,
            start_date=start_date,
            end_date=end_date
        )
        
        # Compile comprehensive results
        backtest_results = {
            'individual_strategies': individual_results,
            'combined_strategy': combined_results,
            'backtest_period': f"{start_date} to {end_date}",
            'symbols_tested': symbols,
            'total_strategies': len(individual_results)
        }
        
        # Generate reports
        self._generate_backtest_reports(backtest_results, backtest_engine)
        
        return backtest_results
    
    def run_live_trading(self):
        """
        Run live trading mode
        """
        self.logger.info("Starting live trading mode")
        self.running = True
        
        try:
            while self.running:
                # Check if market is open (basic check)
                if self._is_market_open():
                    # Update market data
                    self._update_market_data()
                    
                    # Generate and execute signals
                    self._execute_trading_cycle()
                    
                    # Update risk management
                    self._update_risk_management()
                    
                    # Log portfolio status
                    self._log_portfolio_status()
                
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep between cycles
                time.sleep(60)  # 1 minute cycle
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in live trading: {e}")
        finally:
            self.shutdown()
    
    def _schedule_tasks(self):
        """Schedule regular tasks"""
        # Market data updates
        schedule.every(5).minutes.do(self._update_market_data)
        
        # Portfolio rebalancing
        schedule.every().hour.do(self._rebalance_portfolio)
        
        # Daily risk reset
        schedule.every().day.at("09:30").do(self._daily_market_open_tasks)
        schedule.every().day.at("16:00").do(self._daily_market_close_tasks)
        
        # Weekly strategy performance review
        schedule.every().monday.at("06:00").do(self._weekly_performance_review)
        
        self.logger.info("Scheduled tasks configured")
    
    def _update_market_data(self):
        """Update market data for all symbols"""
        try:
            # Fetch real-time data for all symbols
            real_time_data = data_manager.fetch_real_time_data(CONFIG.SYMBOLS)
            
            if real_time_data:
                self.market_data.update(real_time_data)
                self.last_market_data_update = datetime.now()
                self.logger.debug(f"Updated market data for {len(real_time_data)} symbols")
            
            # Also update historical data for strategy analysis
            if not hasattr(self, 'historical_data') or self._should_refresh_historical_data():
                self.historical_data = data_manager.fetch_historical_data(
                    symbols=CONFIG.SYMBOLS,
                    period='1y',
                    interval='1d'
                )
                self.logger.info("Refreshed historical data")
                
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _should_refresh_historical_data(self) -> bool:
        """Check if historical data should be refreshed"""
        if not hasattr(self, 'last_historical_update'):
            self.last_historical_update = datetime.now()
            return True
        
        # Refresh every 4 hours
        return (datetime.now() - self.last_historical_update).seconds > 14400
    
    def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            if not hasattr(self, 'historical_data') or not self.historical_data:
                self.logger.warning("No historical data available for signal generation")
                return
            
            # Generate signals from strategy manager
            signals = strategy_manager.generate_combined_signals(self.historical_data)
            
            if signals:
                self.logger.info(f"Generated {len(signals)} trading signals")
                
                # Extract current prices for execution
                current_prices = {}
                for symbol, data in self.market_data.items():
                    current_prices[symbol] = data.get('price', 0)
                
                # Execute signals
                execution_engine.execute_signals(signals, current_prices)
                
                # Update positions with current prices
                execution_engine.update_positions(current_prices)
                
                # Check stop losses
                execution_engine.check_and_update_stops(current_prices)
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def _update_risk_management(self):
        """Update risk management metrics"""
        try:
            portfolio_summary = execution_engine.get_portfolio_summary()
            
            # Update daily P&L
            current_value = portfolio_summary['total_value']
            if hasattr(self, 'previous_portfolio_value'):
                daily_pnl_change = current_value - self.previous_portfolio_value
                execution_engine.risk_manager.update_daily_pnl(daily_pnl_change)
            
            self.previous_portfolio_value = current_value
            
            # Check risk limits
            if portfolio_summary['unrealized_pnl'] < -CONFIG.MAX_DAILY_LOSS * current_value:
                self.logger.warning("Daily loss limit approached - reducing position sizes")
                # Could implement position size reduction here
            
        except Exception as e:
            self.logger.error(f"Error updating risk management: {e}")
    
    def _log_portfolio_status(self):
        """Log current portfolio status"""
        try:
            summary = execution_engine.get_portfolio_summary()
            
            self.logger.info(
                f"Portfolio Status - Value: ${summary['total_value']:,.2f}, "
                f"Positions: {summary['positions']}, "
                f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f}, "
                f"Daily P&L: ${summary['daily_pnl']:,.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging portfolio status: {e}")
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        # Simple market hours check (9:30 AM - 4:00 PM ET, weekdays)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _daily_market_open_tasks(self):
        """Tasks to run at market open"""
        self.logger.info("Market open - Running daily initialization tasks")
        
        # Reset daily P&L
        execution_engine.risk_manager.reset_daily_pnl()
        
        # Update market data
        self._update_market_data()
        
        # Log opening portfolio status
        self._log_portfolio_status()
    
    def _daily_market_close_tasks(self):
        """Tasks to run at market close"""
        self.logger.info("Market close - Running daily wrap-up tasks")
        
        # Log closing portfolio status
        self._log_portfolio_status()
        
        # Update strategy performance
        executed_trades = execution_engine.executed_orders
        strategy_manager.update_strategy_performance(executed_trades)
        
        # Clear executed orders for next day
        execution_engine.executed_orders = []
    
    def _weekly_performance_review(self):
        """Weekly performance review and strategy adjustment"""
        self.logger.info("Running weekly performance review")
        
        # Get strategy performance summary
        strategy_summary = strategy_manager.get_strategy_summary()
        self.logger.info(f"Strategy weights: {strategy_summary['weights']}")
        
        # Could implement additional weekly tasks here
        # - Rebalance strategy weights
        # - Review and adjust risk parameters
        # - Generate performance reports
    
    def _rebalance_portfolio(self):
        """Rebalance portfolio if needed"""
        try:
            portfolio_summary = execution_engine.get_portfolio_summary()
            
            # Check for concentration risk
            for symbol, details in portfolio_summary['position_details'].items():
                if details['weight'] > CONFIG.MAX_POSITION_SIZE * 1.2:  # 20% buffer
                    self.logger.warning(f"Position {symbol} exceeds size limit: {details['weight']:.2%}")
                    # Could implement automatic rebalancing here
            
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {e}")
    
    def _generate_backtest_reports(self, results: Dict, backtest_engine: BacktestEngine):
        """Generate comprehensive backtest reports"""
        os.makedirs('reports', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate text report for combined strategy
        combined_results = results['combined_strategy']
        report_text = backtest_engine.generate_report(combined_results)
        
        with open(f'reports/backtest_report_{timestamp}.txt', 'w') as f:
            f.write(report_text)
        
        # Generate CSV with individual strategy results
        strategy_comparison = []
        for strategy_name, strategy_results in results['individual_strategies'].items():
            strategy_comparison.append({
                'Strategy': strategy_name,
                'Total Return': strategy_results['total_return'],
                'Annualized Return': strategy_results['annualized_return'],
                'Volatility': strategy_results['volatility'],
                'Sharpe Ratio': strategy_results['sharpe_ratio'],
                'Max Drawdown': strategy_results['max_drawdown'],
                'Win Rate': strategy_results['win_rate'],
                'Total Trades': strategy_results['total_trades']
            })
        
        df_comparison = pd.DataFrame(strategy_comparison)
        df_comparison.to_csv(f'reports/strategy_comparison_{timestamp}.csv', index=False)
        
        self.logger.info(f"Backtest reports saved to reports/ directory")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def shutdown(self):
        """Gracefully shutdown the trading bot"""
        self.logger.info("Shutting down Trading Bot")
        
        try:
            # Stop execution engine
            execution_engine.shutdown()
            
            # Save final portfolio state
            portfolio_summary = execution_engine.get_portfolio_summary()
            
            # Log final status
            self.logger.info(f"Final portfolio value: ${portfolio_summary['total_value']:,.2f}")
            self.logger.info("Trading Bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Algorithmic Trading Bot')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--config', help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize trading bot
    bot = TradingBot(mode=args.mode)
    bot.initialize()
    
    if args.mode == 'backtest':
        # Run backtest
        results = bot.run_backtest(args.start_date, args.end_date)
        print(f"\nBacktest completed!")
        print(f"Combined strategy total return: {results['combined_strategy']['total_return']:.2%}")
        print(f"Combined strategy Sharpe ratio: {results['combined_strategy']['sharpe_ratio']:.3f}")
        print(f"Reports saved to reports/ directory")
        
    else:
        # Run live/paper trading
        try:
            bot.run_live_trading()
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        finally:
            bot.shutdown()

if __name__ == "__main__":
    main()
