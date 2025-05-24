"""
Logging configuration for the algorithmic trading bot.
Provides structured logging with different levels and handlers.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_dir: str = "logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (None for auto-generated)
        log_dir: Directory for log files
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"trading_bot_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)  # More detailed logging to file
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_log_path = os.path.join(log_dir, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path,
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Create specialized loggers
    setup_specialized_loggers(log_dir, detailed_formatter, max_file_size, backup_count)
    
    logging.info(f"Logging initialized - Level: {log_level}, File: {log_path}")
    return root_logger


def setup_specialized_loggers(log_dir: str,
                            formatter: logging.Formatter,
                            max_file_size: int,
                            backup_count: int):
    """Set up specialized loggers for different components."""
    
    # Trading logger
    trading_logger = logging.getLogger('trading')
    trading_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'trading.log'),
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    trading_handler.setFormatter(formatter)
    trading_logger.addHandler(trading_handler)
    trading_logger.setLevel(logging.INFO)
    
    # Strategy logger
    strategy_logger = logging.getLogger('strategy')
    strategy_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'strategy.log'),
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    strategy_handler.setFormatter(formatter)
    strategy_logger.addHandler(strategy_handler)
    strategy_logger.setLevel(logging.INFO)
    
    # Risk logger
    risk_logger = logging.getLogger('risk')
    risk_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'risk.log'),
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    risk_handler.setFormatter(formatter)
    risk_logger.addHandler(risk_handler)
    risk_logger.setLevel(logging.WARNING)
    
    # Data logger
    data_logger = logging.getLogger('data')
    data_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'data.log'),
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    data_handler.setFormatter(formatter)
    data_logger.addHandler(data_handler)
    data_logger.setLevel(logging.INFO)
    
    # Performance logger
    performance_logger = logging.getLogger('performance')
    performance_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'performance.log'),
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    performance_handler.setFormatter(formatter)
    performance_logger.addHandler(performance_handler)
    performance_logger.setLevel(logging.INFO)


class TradingLoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for trading-specific context."""
    
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['strategy'], msg), kwargs


def get_strategy_logger(strategy_name: str) -> TradingLoggerAdapter:
    """
    Get a logger for a specific strategy.
    
    Args:
        strategy_name: Name of the trading strategy
        
    Returns:
        Configured logger adapter
    """
    base_logger = logging.getLogger('strategy')
    return TradingLoggerAdapter(base_logger, {'strategy': strategy_name})


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
    
    def log_execution_time(self, operation: str, execution_time: float):
        """Log execution time for operations."""
        self.logger.info(f"{operation} executed in {execution_time:.4f} seconds")
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage for operations."""
        self.logger.info(f"{operation} memory usage: {memory_mb:.2f} MB")
    
    def log_api_call(self, endpoint: str, response_time: float, status_code: int):
        """Log API call metrics."""
        self.logger.info(f"API call to {endpoint} - Response time: {response_time:.3f}s, Status: {status_code}")


class RiskLogger:
    """Logger for risk-related events and alerts."""
    
    def __init__(self):
        self.logger = logging.getLogger('risk')
    
    def log_risk_alert(self, alert_type: str, message: str, severity: str = "WARNING"):
        """Log risk alerts."""
        level = getattr(logging, severity.upper(), logging.WARNING)
        self.logger.log(level, f"RISK ALERT [{alert_type}]: {message}")
    
    def log_limit_breach(self, limit_type: str, current_value: float, limit_value: float):
        """Log risk limit breaches."""
        self.logger.warning(f"LIMIT BREACH [{limit_type}]: Current {current_value:.4f} > Limit {limit_value:.4f}")
    
    def log_position_change(self, symbol: str, old_position: float, new_position: float):
        """Log position changes."""
        self.logger.info(f"POSITION CHANGE [{symbol}]: {old_position:.2f} -> {new_position:.2f}")


class TradeLogger:
    """Logger for trade execution and order management."""
    
    def __init__(self):
        self.logger = logging.getLogger('trading')
    
    def log_order(self, action: str, symbol: str, quantity: int, price: float, order_type: str = "MARKET"):
        """Log order placement."""
        self.logger.info(f"ORDER {action} [{symbol}]: {quantity} shares at {price:.2f} ({order_type})")
    
    def log_fill(self, symbol: str, quantity: int, fill_price: float, commission: float = 0):
        """Log order fills."""
        self.logger.info(f"FILL [{symbol}]: {quantity} shares at {fill_price:.2f}, Commission: {commission:.2f}")
    
    def log_pnl(self, symbol: str, realized_pnl: float, unrealized_pnl: float):
        """Log P&L updates."""
        self.logger.info(f"PNL [{symbol}]: Realized {realized_pnl:.2f}, Unrealized {unrealized_pnl:.2f}")


def configure_third_party_loggers():
    """Configure logging for third-party libraries."""
    # Reduce verbosity of third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('alpaca_trade_api').setLevel(logging.INFO)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)


def log_system_info():
    """Log system information at startup."""
    import platform
    import psutil
    
    logger = logging.getLogger(__name__)
    
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    logger.info("=== END SYSTEM INFO ===")


# Example usage functions
def example_logging_usage():
    """Example of how to use the logging system."""
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Configure third-party loggers
    configure_third_party_loggers()
    
    # Log system info
    log_system_info()
    
    # Use specialized loggers
    strategy_logger = get_strategy_logger("MeanReversion")
    strategy_logger.info("Strategy initialized successfully")
    
    perf_logger = PerformanceLogger()
    perf_logger.log_execution_time("Data Download", 2.5)
    
    risk_logger = RiskLogger()
    risk_logger.log_risk_alert("HIGH_VOLATILITY", "Market volatility exceeded 30%")
    
    trade_logger = TradeLogger()
    trade_logger.log_order("BUY", "AAPL", 100, 150.0)


if __name__ == "__main__":
    example_logging_usage()
