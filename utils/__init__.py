"""
Utilities package for the algorithmic trading bot.
Provides common helper functions, decorators, and utility classes.
"""

from .helpers import *
from .decorators import *
from .logging_config import setup_logging
from .notifications import EmailNotifier, NotificationManager

__all__ = [
    # Helper functions
    'calculate_returns',
    'calculate_sharpe_ratio', 
    'calculate_max_drawdown',
    'calculate_volatility',
    'calculate_var',
    'calculate_cvar',
    'validate_data',
    'resample_data',
    'clean_data',
    'get_trading_days',
    'is_market_open',
    'next_trading_day',
    'business_days_between',
    
    # Decorators
    'timing_decorator',
    'retry_on_failure',
    'rate_limit',
    'cache_result',
    'validate_market_hours',
    'log_trade_execution',
    'handle_api_errors',
    
    # Logging
    'setup_logging',
    
    # Notifications
    'EmailNotifier',
    'NotificationManager',
]
