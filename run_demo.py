#!/usr/bin/env python3
"""
Demo Trading Bot Launcher - Uses Yahoo Finance data
Perfect for testing strategies before connecting to real brokers like Interactive Brokers
"""

import os
import sys
import time
import signal
import numpy as np
import argparse
import logging
import traceback
import json
import threading
import pandas as pd
import pytz
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from collections import defaultdict, deque

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from execution.demo_trader import DemoTradingEngine
from data.data_manager import DataManager
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.ml_ensemble import MLEnsembleStrategy
from strategies.vwap_strategy import VWAPStrategy
from strategies.bollinger_squeeze import BollingerSqueezeStrategy
from strategies.ichimoku_strategy import IchimokuStrategy
from strategies.support_resistance import SupportResistanceStrategy
from strategies.volume_profile import VolumeProfileStrategy
from strategies.market_microstructure import MarketMicrostructureStrategy
from strategies.gap_trading import GapTradingStrategy
from config.config import (MEAN_REVERSION_CONFIG, MOMENTUM_CONFIG, ML_CONFIG,
                          VWAP_CONFIG, BOLLINGER_SQUEEZE_CONFIG, ICHIMOKU_CONFIG,
                          SUPPORT_RESISTANCE_CONFIG, VOLUME_PROFILE_CONFIG, 
                          MARKET_MICROSTRUCTURE_CONFIG, GAP_TRADING_CONFIG,
                          TIME_BASED_CONFIG)
import logging

# Setup logging with smart message deduplication
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartLogger:
    """Smart logging system that prevents spam and integrates with dashboard"""
    
    def __init__(self, dashboard_url="http://localhost:5001", enable_dashboard=True):
        self.dashboard_url = dashboard_url
        self.enable_dashboard = enable_dashboard
        self.recent_messages = deque(maxlen=100)  # Keep last 100 messages
        self.message_counts = defaultdict(int)
        self.last_sent_time = defaultdict(float)
        self.min_interval = 5  # Minimum seconds between duplicate messages
        
    def _should_log_message(self, message: str, level: str) -> bool:
        """Determine if a message should be logged based on recent activity"""
        current_time = time.time()
        message_key = f"{level}:{message}"
        
        # Always log errors and warnings
        if level in ['ERROR', 'WARNING']:
            return True
            
        # Check if we've seen this exact message recently
        if message_key in self.last_sent_time:
            time_since_last = current_time - self.last_sent_time[message_key]
            if time_since_last < self.min_interval:
                self.message_counts[message_key] += 1
                return False
        
        # Update tracking
        self.last_sent_time[message_key] = current_time
        self.message_counts[message_key] = 1
        return True
    
    def _format_message_with_count(self, message: str, level: str) -> str:
        """Add count information to repeated messages"""
        message_key = f"{level}:{message}"
        count = self.message_counts.get(message_key, 1)
        
        if count > 1:
            return f"{message} (repeated {count}x)"
        return message
    
    def _send_to_dashboard(self, message: str, level: str, msg_type: str = 'general'):
        """Send log message to dashboard if available"""
        if not self.enable_dashboard:
            return
            
        try:
            payload = {
                'level': level,
                'message': message,
                'type': msg_type,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to send to dashboard (non-blocking)
            threading.Thread(
                target=self._post_to_dashboard,
                args=(payload,),
                daemon=True
            ).start()
            
        except Exception:
            pass  # Silently fail dashboard integration
    
    def _post_to_dashboard(self, payload: dict):
        """Post message to dashboard endpoint"""
        try:
            response = requests.post(
                f"{self.dashboard_url}/api/log",
                json=payload,
                timeout=1  # Quick timeout
            )
        except Exception:
            pass  # Silently fail
    
    def info(self, message: str, msg_type: str = 'general'):
        """Log info message with smart filtering"""
        if self._should_log_message(message, 'INFO'):
            formatted_msg = self._format_message_with_count(message, 'INFO')
            logger.info(formatted_msg)
            self._send_to_dashboard(formatted_msg, 'INFO', msg_type)
    
    def warning(self, message: str, msg_type: str = 'general'):
        """Log warning message"""
        formatted_msg = self._format_message_with_count(message, 'WARNING')
        logger.warning(formatted_msg)
        self._send_to_dashboard(formatted_msg, 'WARNING', msg_type)
    
    def error(self, message: str, msg_type: str = 'general'):
        """Log error message"""
        formatted_msg = self._format_message_with_count(message, 'ERROR')
        logger.error(formatted_msg)
        self._send_to_dashboard(formatted_msg, 'ERROR', msg_type)
    
    def trade(self, message: str):
        """Log trade-specific message"""
        # Replace emoji with simple text to avoid Unicode encoding issues
        clean_message = message.replace("[TRADE]", "[TRADE]").replace("[SUCCESS]", "[SUCCESS]").replace("[ERROR]", "[ERROR]")
        self.info(clean_message, 'trade')
    
    def signal(self, message: str):
        """Log signal-specific message"""
        # Replace emoji with simple text to avoid Unicode encoding issues  
        clean_message = message.replace("[SIGNAL]", "[SIGNAL]").replace("[ANALYSIS]", "[ANALYSIS]").replace("[ALERT]", "[ALERT]").replace("[DATA]", "[DATA]")
        self.info(clean_message, 'signal')
    
    def market(self, message: str):
        """Log market data message"""
        # Replace emoji with simple text to avoid Unicode encoding issues
        clean_message = message.replace("[MARKET]", "[MARKET]").replace("[SUCCESS]", "[SUCCESS]").replace("[ERROR]", "[ERROR]").replace("[STOCK]", "[STOCK]").replace("[CRYPTO]", "[CRYPTO]")
        self.info(clean_message, 'market')

# Initialize smart logger
smart_logger = SmartLogger()

class DashboardIntegration:
    """Integration with real-time dashboard"""
    
    def __init__(self, dashboard_url: str = "http://localhost:5000", enabled: bool = True):
        self.dashboard_url = dashboard_url
        self.enabled = enabled
        
        # Test connection to dashboard
        if enabled:
            try:
                response = requests.get(f"{dashboard_url}/api/status", timeout=2)
                if response.status_code == 200:
                    print(f"[DASHBOARD] Connected to dashboard at {dashboard_url}")
                else:
                    print(f"[DASHBOARD] Dashboard not responding, updates disabled")
                    self.enabled = False
            except Exception:
                print(f"[DASHBOARD] Dashboard not available at {dashboard_url}, updates disabled")
                self.enabled = False
    
    def update_portfolio(self, portfolio_data: dict):
        """Send portfolio update to dashboard"""
        if not self.enabled:
            return
            
        try:
            # Format data for dashboard
            dashboard_data = {
                'total_value': portfolio_data.get('total_equity', 0),
                'cash': portfolio_data.get('cash', 0),
                'positions_value': portfolio_data.get('positions_value', 0),
                'positions': portfolio_data.get('num_positions', 0),
                'unrealized_pnl': portfolio_data.get('unrealized_pnl', 0),
                'daily_pnl': portfolio_data.get('unrealized_pnl', 0),  # Use unrealized as daily for demo
                'total_return': portfolio_data.get('total_return', 0),
                'initial_capital': portfolio_data.get('initial_capital', 500),
                'position_details': {},
                'timestamp': datetime.now().isoformat(),
                'is_valid': True
            }
            
            # Add position details
            if 'positions' in portfolio_data and isinstance(portfolio_data['positions'], dict):
                for symbol, pos in portfolio_data['positions'].items():
                    dashboard_data['position_details'][symbol] = {
                        'shares': pos.get('shares', 0),
                        'entry_price': pos.get('entry_price', 0),
                        'current_price': pos.get('current_price', 0),
                        'market_value': pos.get('market_value', 0),
                        'unrealized_pnl': pos.get('unrealized_pnl', 0),
                        'unrealized_pnl_pct': pos.get('unrealized_pnl_pct', 0)
                    }
            
            # Send update to dashboard
            threading.Thread(
                target=self._post_portfolio_update,
                args=(dashboard_data,),
                daemon=True
            ).start()
            
        except Exception as e:
            print(f"[DASHBOARD] Error formatting portfolio data: {e}")
    
    def _post_portfolio_update(self, data: dict):
        """Post portfolio update to dashboard"""
        try:
            response = requests.post(
                f"{self.dashboard_url}/api/portfolio",
                json=data,
                timeout=5
            )
            if response.status_code == 200:
                print(f"[DASHBOARD] Portfolio update sent successfully")
            else:
                print(f"[DASHBOARD] Failed to send portfolio update: {response.status_code}")
        except Exception as e:
            print(f"[DASHBOARD] Error sending portfolio update: {e}")
    
    def send_trade_signal(self, signal_data: dict):
        """Send trading signal to dashboard"""
        if not self.enabled:
            return
            
        try:
            # Send signal to dashboard for activity feed
            threading.Thread(
                target=self._post_signal,
                args=(signal_data,),
                daemon=True
            ).start()
        except Exception:
            pass
    
    def _post_signal(self, data: dict):
        """Post signal to dashboard"""
        try:
            requests.post(
                f"{self.dashboard_url}/api/signal",
                json=data,
                timeout=2
            )
        except Exception:
            pass
    
    def send_trade_execution(self, trade_data: dict):
        """Send trade execution to dashboard"""
        if not self.enabled:
            return
            
        try:
            threading.Thread(
                target=self._post_trade,
                args=(trade_data,),
                daemon=True
            ).start()
        except Exception:
            pass
    
    def _post_trade(self, data: dict):
        """Post trade execution to dashboard"""
        try:
            requests.post(
                f"{self.dashboard_url}/api/trade",
                json=data,
                timeout=2
            )
        except Exception:
            pass

class DemoTradingBot:
    """
    Demo trading bot that uses Yahoo Finance data with real-time dashboard integration
    Can be easily switched to Interactive Brokers later
    """
    
    def __init__(self, initial_capital: float = 500, enable_dashboard: bool = True):
        self.demo_trader = DemoTradingEngine(initial_capital)
        self.data_manager = DataManager()
        self.running = False
        
        # Dashboard integration
        self.dashboard = DashboardIntegration() if enable_dashboard else None
        
        # Initialize strategies - include new advanced strategies for comprehensive testing
        self.strategies = {
            # Core strategies
            'mean_reversion': MeanReversionStrategy(MEAN_REVERSION_CONFIG),
            'momentum': MomentumStrategy(MOMENTUM_CONFIG),
            'ml_ensemble': MLEnsembleStrategy(ML_CONFIG),
            
            # Advanced technical strategies
            'vwap': VWAPStrategy(VWAP_CONFIG),
            'bollinger_squeeze': BollingerSqueezeStrategy(BOLLINGER_SQUEEZE_CONFIG),
            'ichimoku': IchimokuStrategy(ICHIMOKU_CONFIG),
            'support_resistance': SupportResistanceStrategy(SUPPORT_RESISTANCE_CONFIG),
            
            # Volume-based strategies
            'volume_profile': VolumeProfileStrategy(VOLUME_PROFILE_CONFIG),
            
            # Market microstructure analysis
            'market_microstructure': MarketMicrostructureStrategy(MARKET_MICROSTRUCTURE_CONFIG),
            
            # Gap trading for overnight opportunities
            'gap_trading': GapTradingStrategy(GAP_TRADING_CONFIG)
        }
        
        # Trading symbols (expanded for 24/7 testing and small capital trading)
        self.symbols = [
            # Major Tech Stocks (fractional shares available)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'AMD', 'NVDA', 'TSM', 'INTC', 'CRM',
            
            # Canadian Stocks (affordable pricing)
            'SHOP.TO', 'TD.TO', 'RY.TO', 'BAM.TO', 'CNR.TO',
            
            # ETFs (diversified exposure)
            'SPY', 'QQQ', 'VTI', 'IWM', 'EEM',
            
            # Affordable Stocks (under $50)
            'F', 'GE', 'T', 'BAC', 'WFC', 'PFE', 'XOM', 'KO',
            
            # Small Cap Stocks (under $20 - perfect for $500 capital)
            'NOK', 'SIRI', 'BBD-B.TO', 'VALE', 'ITUB', 'GOLD',
            'SNAP', 'PLTR', 'WISH', 'BB', 'AMC', 'NIO',
            
            # Cryptocurrency ETFs/Stocks (24/7 crypto exposure)
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD',       # Direct crypto
            'COIN', 'MSTR', 'RIOT', 'MARA',                   # Crypto-related stocks
            'ARKK', 'ARKF',                                   # Innovation ETFs with crypto exposure
            
            # Penny Stocks & Micro-caps (high volatility for testing)
            'CTRM', 'TOPS', 'SHIP', 'GNUS', 'XELA', 'EXPR'
        ]
        
        # Benefits of this expanded universe:
        # 1. Cryptocurrencies trade 24/7 - perfect for weekend/evening testing
        # 2. Small cap stocks under $20 - affordable for $500 capital
        # 3. Mix of volatility levels - from stable ETFs to volatile penny stocks
        # 4. International exposure - Canadian stocks for currency diversification
        # 5. Different market caps - from micro-caps to mega-caps for strategy testing
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Strategy allocation (what % of capital to allocate to each strategy)
        self.strategy_allocation = {
            # Core strategies (higher allocation)
            'mean_reversion': 0.20,     # 20%
            'momentum': 0.20,           # 20%
            'ml_ensemble': 0.15,        # 15%
            
            # Advanced technical strategies (moderate allocation)
            'vwap': 0.10,               # 10%
            'bollinger_squeeze': 0.08,  # 8%
            'ichimoku': 0.08,           # 8%
            'support_resistance': 0.08, # 8%
            
            # Specialized strategies (smaller allocation)
            'volume_profile': 0.04,     # 4%
            'market_microstructure': 0.04, # 4%
            'gap_trading': 0.03         # 3%
        }
        # Total allocation: 100%
        
        # Maximum position per symbol (diversification limit)
        self.max_position_pct = 0.15  # Maximum 15% of portfolio in any single symbol
        
        # Crypto symbols for aftermarket trading
        self.crypto_symbols = {
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD', 
            'AVAX-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XRP-USD'
        }
        
        # Market hours (Eastern Time)
        self.market_open_time = (9, 30)  # 9:30 AM
        self.market_close_time = (16, 0)  # 4:00 PM
        self.last_position_age_check = datetime.now()
        
        # Time-based configuration
        self.time_config = TIME_BASED_CONFIG
        
        # Adaptive polling configuration
        self.polling_config = {
            'min_interval': 15,      # Minimum 15 seconds between cycles
            'max_interval': 300,     # Maximum 5 minutes between cycles
            'default_interval': 60,  # Default 1 minute
            'signal_boost_factor': 0.3,   # Speed up by 70% when signals detected
            'volatility_boost_factor': 0.5, # Speed up by 50% during high volatility
            'market_hours_factor': 0.7,     # Faster during market hours
            'crypto_hours_factor': 1.2,     # Slower during crypto-only hours
            'no_signal_slowdown': 1.5       # Slow down by 50% when no signals
        }
        
        # Tracking for adaptive polling
        self.recent_signals = []  # Track recent signal history
        self.recent_trades = []   # Track recent trade history
        self.last_cycle_time = None
        self.consecutive_no_signals = 0
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n[STOP] Received shutdown signal ({signum}), stopping bot...")
        self.running = False
    
    def is_market_hours(self):
        """Check if US stock market is currently open"""
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now_et.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if within market hours (9:30 AM - 4:00 PM ET)
        market_open = now_et.replace(hour=self.market_open_time[0], minute=self.market_open_time[1], second=0, microsecond=0)
        market_close = now_et.replace(hour=self.market_close_time[0], minute=self.market_close_time[1], second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    
    def get_preferred_symbols(self):
        """Get preferred symbols based on market hours and positions"""
        current_positions = self.demo_trader.get_portfolio_summary()['positions']
        has_positions = len(current_positions) > 0
        is_market_open = self.is_market_hours()
        
        if not is_market_open and not has_positions:
            # Aftermarket hours with no positions - prefer crypto
            crypto_list = list(self.crypto_symbols)
            other_symbols = [s for s in self.symbols if s not in self.crypto_symbols]
            preferred_symbols = crypto_list + other_symbols[:10]  # Add some regular stocks too
            print(f"[MOON] Aftermarket hours detected - preferring crypto trading")
        else:
            # Regular market hours or have existing positions
            preferred_symbols = self.symbols
            if is_market_open:
                print(f"[SUN] Market hours detected - using full symbol list")
            else:
                print(f"[MOON] Aftermarket hours but have positions - using all symbols")
        
        return preferred_symbols
    
    def calculate_market_volatility(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current market volatility for adaptive polling"""
        volatilities = []
        
        for symbol, data in market_data.items():
            if len(data) >= 20:
                # Calculate 20-period volatility
                returns = data['Close'].pct_change().dropna()
                if len(returns) >= 10:
                    volatility = returns.rolling(window=10).std().iloc[-1] * 100
                    if not pd.isna(volatility):
                        volatilities.append(volatility)
        
        if volatilities:
            avg_volatility = np.mean(volatilities)
            # Normalize to 0-1 scale (assume 5% daily volatility is "high")
            return min(avg_volatility / 5.0, 1.0)
        
        return 0.5  # Default moderate volatility
    
    def calculate_next_polling_interval(self, signals: List[Dict], market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate adaptive polling interval based on market conditions"""
        base_interval = self.polling_config['default_interval']
        
        # Factor 1: Signal activity
        signal_factor = 1.0
        if signals:
            # Speed up when signals are detected
            signal_factor = self.polling_config['signal_boost_factor']
            self.consecutive_no_signals = 0
            self.recent_signals.extend(signals)
        else:
            # Slow down when no signals
            self.consecutive_no_signals += 1
            if self.consecutive_no_signals >= 3:
                signal_factor = self.polling_config['no_signal_slowdown']
        
        # Keep only recent signals (last 10 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.recent_signals = [s for s in self.recent_signals 
                              if s.get('timestamp', datetime.now()) > cutoff_time]
        
        # Factor 2: Market volatility
        volatility = self.calculate_market_volatility(market_data)
        if volatility > 0.7:  # High volatility
            volatility_factor = self.polling_config['volatility_boost_factor']
        else:
            volatility_factor = 1.0
        
        # Factor 3: Market hours
        is_market_open = self.is_market_hours()
        if is_market_open:
            time_factor = self.polling_config['market_hours_factor']
        else:
            # Check if we're mainly trading crypto
            crypto_signals = len([s for s in signals if s.get('symbol', '') in self.crypto_symbols])
            if crypto_signals > len(signals) * 0.5:  # Majority crypto signals
                time_factor = self.polling_config['crypto_hours_factor']
            else:
                time_factor = 1.0
        
        # Factor 4: Recent trade activity
        recent_trades = self.demo_trader.get_recent_trades(limit=5)
        trade_factor = 1.0
        if recent_trades:
            last_trade_time = datetime.fromisoformat(recent_trades[-1]['timestamp'])
            minutes_since_trade = (datetime.now() - last_trade_time).total_seconds() / 60
            if minutes_since_trade < 5:  # Recent trade in last 5 minutes
                trade_factor = 0.5  # Speed up significantly
            elif minutes_since_trade < 15:  # Trade in last 15 minutes
                trade_factor = 0.7  # Speed up moderately
        
        # Calculate final interval
        final_interval = base_interval * signal_factor * volatility_factor * time_factor * trade_factor
        
        # Apply min/max bounds
        final_interval = max(self.polling_config['min_interval'], 
                           min(self.polling_config['max_interval'], final_interval))
        
        # Log the decision for transparency
        factors_str = f"signal:{signal_factor:.2f} vol:{volatility_factor:.2f} time:{time_factor:.2f} trade:{trade_factor:.2f}"
        print(f"[CLOCK] Next cycle in {final_interval:.0f}s (factors: {factors_str}, volatility: {volatility:.2f})")
        
        return final_interval
    
    def check_diversification_limit(self, symbol, position_value):
        """Check if adding this position would exceed diversification limits"""
        portfolio_value = self.demo_trader.get_portfolio_summary()['total_equity']
        current_position = self.demo_trader.get_position(symbol)
        
        # Calculate what the total position value would be
        current_position_value = 0
        if current_position:
            current_position_value = current_position.shares * current_position.current_price
        
        total_position_value = current_position_value + position_value
        position_percentage = total_position_value / portfolio_value
        
        if position_percentage > self.max_position_pct:
            return False, f"Would exceed {self.max_position_pct*100:.0f}% limit ({position_percentage*100:.1f}%)"
        
        return True, ""
    
    def check_position_ages(self):
        """Check and force close positions that have exceeded maximum holding period"""
        current_time = datetime.now()
        
        # Only check every hour to avoid excessive processing
        time_since_last_check = current_time - self.last_position_age_check
        if time_since_last_check.total_seconds() < self.time_config['position_age_check_interval']:
            return
        
        self.last_position_age_check = current_time
        
        # Check for aged positions
        max_age_days = self.time_config['max_holding_period_days']
        aged_positions = self.demo_trader.get_aged_positions(max_age_days)
        
        if aged_positions:
            smart_logger.warning(f"[CLOCK] Found {len(aged_positions)} positions exceeding {max_age_days} day holding limit", 'risk')
            
            for symbol in aged_positions:
                holding_time = self.demo_trader.get_position_holding_time(symbol)
                smart_logger.warning(f"[CLOCK] {symbol}: held for {holding_time.days} days, {holding_time.seconds//3600} hours", 'risk')
            
            # Force close aged positions
            closed_positions = self.demo_trader.force_close_aged_positions(max_age_days)
            
            if closed_positions:
                smart_logger.info(f"[SUCCESS] Closed {len(closed_positions)} aged positions: {', '.join(closed_positions)}", 'risk')
    
    def check_time_based_risks(self):
        """Check time-based risks including position ages and market hours compliance"""
        try:
            portfolio_summary = self.demo_trader.get_portfolio_summary()
            
            if not portfolio_summary['positions']:
                return
            
            # Check for positions approaching or exceeding max holding period
            max_age_days = self.time_config['max_holding_period_days']
            warning_threshold = max_age_days * 0.8  # Warn at 80% of max holding period
            
            aged_positions = []
            warning_positions = []
            
            for symbol, pos_data in portfolio_summary['positions'].items():
                entry_time = datetime.fromisoformat(pos_data['entry_time'])
                holding_time = datetime.now() - entry_time
                holding_days = holding_time.days + (holding_time.seconds / 86400)  # Include hours as fractional days
                
                if holding_days > max_age_days:
                    aged_positions.append((symbol, holding_days))
                elif holding_days > warning_threshold:
                    warning_positions.append((symbol, holding_days))
            
            # Display warnings for positions approaching limit
            if warning_positions:
                smart_logger.warning(f"[WARNING] {len(warning_positions)} positions approaching {max_age_days}-day holding limit:", 'risk')
                for symbol, days in warning_positions:
                    smart_logger.warning(f"   [CALENDAR] {symbol}: {days:.1f} days (limit: {max_age_days} days)", 'risk')
            
            # Display alerts for positions exceeding limit (these will be auto-closed)
            if aged_positions:
                smart_logger.warning(f"[ALERT] {len(aged_positions)} positions exceeded {max_age_days}-day holding limit:", 'risk')
                for symbol, days in aged_positions:
                    smart_logger.warning(f"   [CLOCK] {symbol}: {days:.1f} days - will be auto-closed", 'risk')
                    
        except Exception as e:
            smart_logger.error(f"Error in time-based risk check: {e}", 'risk')

    def display_banner(self):
        """Display welcome banner"""
        print("\n" + "="*70)
        print("[ROBOT] ALGORITHMIC TRADING BOT - DEMO MODE")
        print("[CHART] Using Yahoo Finance Real-Time Data")
        print("[FLAG] Ready for Interactive Brokers Integration Later")
        print("="*70)
        print(f"[MONEY] Initial Capital: ${self.demo_trader.initial_capital:,.2f}")
        print(f"[CHART] Trading Universe: {len(self.symbols)} symbols")
        print(f"[BRAIN] Active Strategies: {', '.join(self.strategies.keys())}")
        print(f"[CLOCK] Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    
    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch current market data for preferred symbols"""
        smart_logger.market("[CHART] Fetching market data...")
        
        # Get preferred symbols based on market hours and positions
        preferred_symbols = self.get_preferred_symbols()
        
        market_data = {}
        
        for symbol in preferred_symbols:
            try:
                # Get recent historical data for strategy analysis
                # Increased from 30d to 100d to support Ichimoku (needs 52+ periods) and other advanced strategies
                data = self.data_manager.get_historical_data(
                    [symbol], 
                    period='365d'  # 100 days of data for advanced technical analysis
                )
                
                if symbol in data and not data[symbol].empty:
                    market_data[symbol] = data[symbol]
                    current_price = data[symbol]['Close'].iloc[-1]
                    crypto_indicator = "[CRYPTO]" if symbol in self.crypto_symbols else "[STOCK]"
                    smart_logger.market(f"[SUCCESS] {crypto_indicator} {symbol}: ${current_price:.2f}")
                else:
                    smart_logger.warning(f"[ERROR] {symbol}: No data available", 'market')
                    
            except Exception as e:
                smart_logger.error(f"[ERROR] {symbol}: Error - {e}", 'market')
        
        smart_logger.market(f"[CHART] Successfully fetched data for {len(market_data)} symbols")
        return market_data
    
    def generate_trading_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate trading signals from all strategies"""
        smart_logger.signal("[BRAIN] Generating trading signals...")
        all_signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                smart_logger.signal(f"[SEARCH] Running {strategy_name} strategy...")
                strategy_signals_count = 0
                
                # Generate signals for each symbol
                for symbol, data in market_data.items():
                    if len(data) < 20:  # Need enough data for analysis
                        continue
                    
                    # Each strategy generates its own signals
                    signal = strategy.generate_signal(data)
                    
                    # Process signals with smart logging
                    if signal is None:
                        # Don't spam with "no signal" messages - smart logger will filter
                        pass
                    elif signal.get('action') == 'hold' or signal.get('confidence', 0) < 0.1:
                        # Don't spam with hold signals
                        pass
                    elif not signal.get('action') in ['buy', 'sell']:
                        action = signal.get('action', 'None')
                        confidence = signal.get('confidence', 0.0)
                        smart_logger.warning(f"[SEARCH] {symbol}: Invalid action '{action}' (confidence: {confidence:.2f})", 'signal')
                    else:
                        signal['symbol'] = symbol
                        signal['strategy'] = strategy_name
                        signal['timestamp'] = datetime.now()
                        signal['price'] = data['Close'].iloc[-1]
                        all_signals.append(signal)
                        strategy_signals_count += 1
                        
                        action = signal['action'].upper()
                        confidence = signal.get('confidence', 0.5)
                        signal_msg = f"[ALERT] {symbol}: {action} (confidence: {confidence:.2f})"
                        smart_logger.signal(signal_msg)
                        
                        # Send to dashboard
                        if self.dashboard:
                            self.dashboard.send_trade_signal({
                                'symbol': symbol,
                                'action': action,
                                'confidence': confidence,
                                'strategy': strategy_name,
                                'price': signal['price'],
                                'timestamp': signal['timestamp'].isoformat()
                            })
                
                if strategy_signals_count > 0:
                    smart_logger.signal(f"[CHART] {strategy_name}: Generated {strategy_signals_count} valid signals")
                
            except Exception as e:
                smart_logger.error(f"[ERROR] Error in {strategy_name}: {e}", 'signal')
                import traceback
                smart_logger.error(f"Traceback: {traceback.format_exc()}", 'signal')
        
        smart_logger.signal(f"[BRAIN] Generated {len(all_signals)} trading signals total")
        return all_signals
    
    def execute_trades(self, signals: List[Dict]):
        """Execute trades based on signals with market hours restrictions"""
        if not signals:
            smart_logger.info("[BRIEFCASE] No trades to execute", 'trade')
            return
        
        # Check position ages first and close old positions
        self.check_position_ages()
        
        smart_logger.trade("[BRIEFCASE] Executing trades...")
        portfolio_value = self.demo_trader.get_portfolio_summary()['total_equity']
        is_market_open = self.is_market_hours()
        
        # Filter signals based on market hours restrictions
        valid_signals = []
        for signal in signals:
            symbol = signal['symbol']
            
            # Check if we can trade this symbol at current time
            if self.demo_trader.can_trade_symbol_now(symbol, is_market_open):
                valid_signals.append(signal)
            else:
                smart_logger.warning(f"[ERROR] Cannot trade {symbol} outside market hours (not crypto)", 'trade')
        
        if not valid_signals:
            smart_logger.info("[BRIEFCASE] No valid trades after market hours filtering", 'trade')
            return
        
        # Group signals by strategy for proper allocation
        strategy_signals = {}
        for signal in valid_signals:
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
                    # For small capital accounts, use more aggressive position sizing
                    if portfolio_value <= 1000:  # Small account
                        # Use 25% of strategy capital per position for small accounts
                        position_value = strategy_capital * 0.25
                    else:
                        # Use 15% of strategy capital per position for larger accounts
                        position_value = strategy_capital * 0.15
                    
                    # Check diversification limits
                    can_buy, reason = self.check_diversification_limit(symbol, position_value)
                    if not can_buy:
                        smart_logger.warning(f"[ERROR] Diversification limit: {symbol} - {reason}", 'trade')
                        continue
                    
                    shares = int(position_value / price)
                    
                    # For very small capital, ensure we can buy at least 1 share of reasonably priced stocks
                    if shares == 0 and price <= strategy_capital * 0.5:  # If stock costs less than 50% of strategy capital
                        shares = 1  # Buy 1 share minimum
                        position_value = shares * price
                        
                        # Re-check diversification with minimum position
                        can_buy, reason = self.check_diversification_limit(symbol, position_value)
                        if not can_buy:
                            smart_logger.warning(f"[ERROR] Diversification limit (min position): {symbol} - {reason}", 'trade')
                            continue
                    
                    if shares > 0 and position_value <= self.demo_trader.cash:
                        success = self.demo_trader.place_order(
                            symbol, shares, 'buy', strategy_name
                        )
                        if success:
                            crypto_indicator = "[CRYPTO]" if symbol in self.crypto_symbols else "[STOCK]"
                            trade_msg = f"[SUCCESS] {crypto_indicator} Bought {shares} shares of {symbol} @ ${price:.2f} (Total: ${position_value:.2f})"
                            smart_logger.trade(trade_msg)
                    else:
                        if shares == 0:
                            smart_logger.warning(f"[ERROR] Cannot afford {symbol} @ ${price:.2f} (would need ${price:.2f}, allocated ${position_value:.2f})", 'trade')
                        elif position_value > self.demo_trader.cash:
                            smart_logger.warning(f"[ERROR] Insufficient cash for {symbol}: need ${position_value:.2f}, have ${self.demo_trader.cash:.2f}", 'trade')
                
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
                            trade_msg = f"[SUCCESS] Sold {shares_to_sell} shares of {symbol} @ ${price:.2f}"
                            smart_logger.trade(trade_msg)
                    else:
                        smart_logger.warning(f"[ERROR] No position in {symbol} to sell", 'trade')
        
        smart_logger.trade("[BRIEFCASE] Trade execution completed")
    
    def display_portfolio_status(self):
        """Display current portfolio status and update dashboard"""
        summary = self.demo_trader.get_portfolio_summary()
        
        print("[BRIEFCASE] PORTFOLIO STATUS")
        print("-" * 50)
        print(f"[MONEY] Cash: ${summary['cash']:,.2f}")
        print(f"[CHART] Positions Value: ${summary['positions_value']:,.2f}")
        print(f"[MONEY] Total Equity: ${summary['total_equity']:,.2f}")
        print(f"[CHART] Total Return: {summary['total_return']:+.2f}%")
        print(f"[LIST] Active Positions: {summary['num_positions']}")
        
        if summary['unrealized_pnl'] != 0:
            pnl_color = "[UP]" if summary['unrealized_pnl'] > 0 else "[DOWN]"
            print(f"{pnl_color} Unrealized P&L: ${summary['unrealized_pnl']:+,.2f}")
        
        # Show top positions
        if summary['positions']:
            print(f"\n[LIST] CURRENT POSITIONS:")
            sorted_positions = sorted(
                summary['positions'].items(),
                key=lambda x: x[1]['market_value'],
                reverse=True
            )
            
            for symbol, pos in sorted_positions[:5]:  # Show top 5
                pnl_pct = pos['unrealized_pnl_pct']
                pnl_indicator = "[UP]" if pnl_pct > 0 else "[DOWN]" if pnl_pct < 0 else "[NEUTRAL]"
                print(f"   {pnl_indicator} {symbol}: {pos['shares']:.0f} shares @ ${pos['current_price']:.2f} "
                      f"(P&L: {pnl_pct:+.1f}%)")
        
        print()
        
        # Send portfolio update to dashboard
        if self.dashboard:
            portfolio_data = {
                **summary,
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.demo_trader.get_performance_metrics() or {}
            }
            self.dashboard.update_portfolio(portfolio_data)
    
    def display_recent_activity(self):
        """Display recent trading activity"""
        recent_trades = self.demo_trader.get_recent_trades(limit=5)
        
        if recent_trades:
            print("[LIST] RECENT TRADES")
            print("-" * 50)
            for trade in recent_trades[-5:]:
                trade_time = datetime.fromisoformat(trade['timestamp']).strftime('%H:%M:%S')
                action_icon = "[BUY]" if trade['type'] == 'buy' else "[SELL]"
                print(f"   {action_icon} {trade_time} - {trade['type'].upper()} "
                      f"{trade['shares']:.0f} {trade['symbol']} @ ${trade['price']:.2f} "
                      f"({trade['strategy']})")
            print()
    
    def run_trading_cycle(self):
        """Run one complete trading cycle with adaptive timing tracking"""
        cycle_start = datetime.now()
        print(f"\n{'='*20} TRADING CYCLE {cycle_start.strftime('%H:%M:%S')} {'='*20}")
        
        try:
            # 1. Fetch market data
            market_data = self.fetch_market_data()
            
            if not market_data:
                print("[ERROR] No market data available, skipping cycle\n")
                return []
            
            # 2. Generate trading signals
            signals = self.generate_trading_signals(market_data)
            
            # 3. Execute trades (includes position age checking)
            self.execute_trades(signals)
            
            # 4. Check time-based risks
            self.check_time_based_risks()
            
            # 5. Update and display portfolio
            self.display_portfolio_status()
            self.display_recent_activity()
            
            # 6. Performance metrics
            metrics = self.demo_trader.get_performance_metrics()
            if metrics:
                print("[CHART] PERFORMANCE METRICS")
                print("-" * 50)
                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
                print(f"   Total Trades: {metrics.get('total_trades', 0)}")
                print()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            print(f"[CLOCK] Cycle completed in {cycle_duration:.1f} seconds")
            
            # Calculate next polling interval based on signals and market conditions
            next_interval = self.calculate_next_polling_interval(signals, market_data)
            
            return signals, market_data, next_interval
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            print(f"[ERROR] Error in trading cycle: {e}")
            return [], {}, self.polling_config['default_interval']
    
    def run(self, cycle_interval_minutes: float = 1):
        """
        Main trading loop with adaptive polling
        
        Args:
            cycle_interval_minutes: Initial polling interval (will adapt based on market conditions)
        """
        self.display_banner()
        self.running = True
        cycle_count = 0
        
        # Convert initial interval to seconds for adaptive polling
        current_interval_seconds = cycle_interval_minutes * 60
        
        # Display initial portfolio state
        self.display_portfolio_status()
        
        print(f"[REFRESH] ADAPTIVE POLLING ENABLED")
        print(f"   [CHART] Range: {self.polling_config['min_interval']}s - {self.polling_config['max_interval']}s")
        print(f"   [TARGET] Initial: {current_interval_seconds:.0f}s")
        print(f"   [LIGHTNING] Speed up factors: Signals ({self.polling_config['signal_boost_factor']:.1f}x), Volatility ({self.polling_config['volatility_boost_factor']:.1f}x)")
        print()
        
        try:
            while self.running:
                cycle_count += 1
                
                # Run trading cycle and get adaptive interval
                result = self.run_trading_cycle()
                
                if isinstance(result, tuple) and len(result) == 3:
                    signals, market_data, next_interval = result
                    current_interval_seconds = next_interval
                else:
                    # Fallback if cycle didn't return proper result
                    current_interval_seconds = self.polling_config['default_interval']
                
                if not self.running:
                    break
                
                # Adaptive wait with countdown
                self.adaptive_wait(current_interval_seconds)
        
        except KeyboardInterrupt:
            print("\n[STOP] Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"[ERROR] Unexpected error: {e}")
        finally:
            self.shutdown()
    
    def adaptive_wait(self, wait_seconds: float):
        """
        Intelligent wait with countdown and opportunity monitoring
        """
        total_seconds = int(wait_seconds)
        
        if total_seconds <= 30:
            # Short waits - show every 5 seconds
            countdown_interval = 5
            unit = "seconds"
        elif total_seconds <= 120:
            # Medium waits - show every 15 seconds  
            countdown_interval = 15
            unit = "seconds"
        else:
            # Long waits - show every 30 seconds
            countdown_interval = 30
            unit = "seconds"
        
        print(f"[ALARM] Adaptive wait: {total_seconds} seconds until next cycle...")
        
        for i in range(total_seconds):
            if not self.running:
                break
            
            time.sleep(1)
            
            # Show countdown at intervals
            remaining = total_seconds - i
            if remaining > 0 and remaining % countdown_interval == 0:
                if remaining >= 60:
                    remaining_display = f"{remaining // 60}m {remaining % 60}s"
                else:
                    remaining_display = f"{remaining}s"
                print(f"   [CLOCK] {remaining_display} remaining...")
        
        if self.running and total_seconds > 0:
            print("[REFRESH] Wait complete - starting next cycle")
    
    def shutdown(self):
        """Shutdown the trading bot gracefully"""
        print(f"\n[REFRESH] Shutting down demo trading bot...")
        
        # Display final summary
        final_summary = self.demo_trader.get_portfolio_summary()
        metrics = self.demo_trader.get_performance_metrics()
        
        print("\n" + "="*60)
        print("[CHART] FINAL SESSION SUMMARY")
        print("="*60)
        print(f"[MONEY] Final Equity: ${final_summary['total_equity']:,.2f}")
        print(f"[CHART] Total Return: {final_summary['total_return']:+.2f}%")
        print(f"[MONEY] Cash: ${final_summary['cash']:,.2f}")
        print(f"[LIST] Final Positions: {final_summary['num_positions']}")
        
        if metrics:
            print(f"[CHART] Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"[DOWN] Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"[REFRESH] Total Trades: {metrics.get('total_trades', 0)}")
        
        # Show final positions
        if final_summary['positions']:
            print(f"\n[LIST] FINAL POSITIONS:")
            for symbol, pos in final_summary['positions'].items():
                pnl_pct = pos['unrealized_pnl_pct']
                pnl_indicator = "[UP]" if pnl_pct > 0 else "[DOWN]"
                print(f"   {pnl_indicator} {symbol}: {pos['shares']:.0f} shares "
                      f"(P&L: {pnl_pct:+.1f}%)")
        
        print("\n[SUCCESS] Demo trading session completed!")
        print("[REFRESH] Ready to connect to Interactive Brokers when you're ready!")
        print("="*60)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Trading Bot with Adaptive Polling')
    parser.add_argument('--capital', type=float, default=500,
                       help='Initial capital (default: $500)')
    parser.add_argument('--interval', type=float, default=1,
                       help='Initial polling interval in minutes (default: 1, will adapt automatically)')
    parser.add_argument('--min-interval', type=int, default=15,
                       help='Minimum polling interval in seconds (default: 15)')
    parser.add_argument('--max-interval', type=int, default=300,
                       help='Maximum polling interval in seconds (default: 300)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset portfolio to initial state')
    
    args = parser.parse_args()
    
    print("[ROCKET] Starting Demo Trading Bot with Adaptive Polling...")
    
    # Create bot instance
    bot = DemoTradingBot(args.capital)
    
    # Configure adaptive polling if custom values provided
    if args.min_interval != 15:
        bot.polling_config['min_interval'] = args.min_interval
    if args.max_interval != 300:
        bot.polling_config['max_interval'] = args.max_interval
    
    # Reset portfolio if requested
    if args.reset:
        print("[REFRESH] Resetting portfolio to initial state...")
        bot.demo_trader.reset_portfolio()
    
    # Start the trading bot
    try:
        bot.run(cycle_interval_minutes=args.interval)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"[EXPLOSION] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
