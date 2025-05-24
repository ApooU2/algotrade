"""
Decorators for the algorithmic trading bot.
Provides timing, retry, rate limiting, and other utility decorators.
"""

import time
import functools
import logging
from typing import Callable, Any, Optional, Dict
from datetime import datetime, timedelta
import threading
from collections import defaultdict


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def retry_decorator(max_retries: int = 3,
                   delay: float = 1.0,
                   backoff_factor: float = 2.0,
                   exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Exponential backoff factor
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logging.error(f"{func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logging.warning(f"{func.__name__} attempt {attempt + 1} failed: {str(e)}. Retrying in {current_delay:.2f} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            return None  # Should never reach here
        
        return wrapper
    return decorator


# Alias for backward compatibility
retry_on_failure = retry_decorator


class RateLimiter:
    """Rate limiter for API calls and function execution."""
    
    def __init__(self):
        self.calls = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str, max_calls: int, time_window: int) -> bool:
        """
        Check if a call is allowed under rate limits.
        
        Args:
            key: Unique identifier for the rate limit
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
            
        Returns:
            True if call is allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            cutoff_time = now - time_window
            
            # Remove old calls
            self.calls[key] = [call_time for call_time in self.calls[key] if call_time > cutoff_time]
            
            # Check if new call is allowed
            if len(self.calls[key]) < max_calls:
                self.calls[key].append(now)
                return True
            
            return False


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limit_decorator(max_calls: int = 100,
                        time_window: int = 60,
                        key_func: Optional[Callable] = None) -> Callable:
    """
    Decorator to rate limit function calls.
    
    Args:
        max_calls: Maximum number of calls allowed
        time_window: Time window in seconds
        key_func: Function to generate rate limit key from args/kwargs
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = func.__name__
            
            # Check rate limit
            if not _rate_limiter.is_allowed(key, max_calls, time_window):
                wait_time = time_window / max_calls
                logging.warning(f"Rate limit exceeded for {func.__name__}. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Alias for backward compatibility
rate_limit = rate_limit_decorator


def cache_decorator(ttl: Optional[int] = None) -> Callable:
    """
    Simple caching decorator with optional TTL.
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if cached result exists and is valid
            if key in cache:
                if ttl is None or (time.time() - cache_times[key]) < ttl:
                    return cache[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        
        return wrapper
    return decorator


def market_hours_only(market_open: str = "09:30",
                     market_close: str = "16:00",
                     timezone: str = "US/Eastern") -> Callable:
    """
    Decorator to only execute function during market hours.
    
    Args:
        market_open: Market open time (HH:MM format)
        market_close: Market close time (HH:MM format)
        timezone: Market timezone
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import pytz
            
            # Get current time in market timezone
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
            
            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                logging.warning(f"{func.__name__} not executed - market is closed (weekend)")
                return None
            
            # Parse market hours
            open_time = datetime.strptime(market_open, "%H:%M").time()
            close_time = datetime.strptime(market_close, "%H:%M").time()
            
            current_time = now.time()
            
            # Check if current time is within market hours
            if open_time <= current_time <= close_time:
                return func(*args, **kwargs)
            else:
                logging.warning(f"{func.__name__} not executed - outside market hours ({current_time})")
                return None
        
        return wrapper
    return decorator


def log_exceptions(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator to log exceptions without re-raising them.
    
    Args:
        logger: Custom logger instance
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log = logger or logging.getLogger(__name__)
                log.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                return None
        
        return wrapper
    return decorator


def validate_inputs(**type_checks: type) -> Callable:
    """
    Decorator to validate function input types.
    
    Args:
        **type_checks: Keyword arguments mapping parameter names to expected types
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(f"Parameter '{param_name}' must be of type {expected_type.__name__}, got {type(value).__name__}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def monitor_performance(threshold_seconds: float = 1.0) -> Callable:
    """
    Decorator to monitor function performance and log slow executions.
    
    Args:
        threshold_seconds: Time threshold for logging warnings
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > threshold_seconds:
                logging.warning(f"{func.__name__} took {execution_time:.4f} seconds (threshold: {threshold_seconds}s)")
            
            return result
        
        return wrapper
    return decorator


def singleton(cls):
    """
    Decorator to create singleton classes.
    
    Args:
        cls: Class to make singleton
        
    Returns:
        Singleton class
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def async_retry(max_retries: int = 3,
               delay: float = 1.0,
               backoff_factor: float = 2.0) -> Callable:
    """
    Async version of retry decorator.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Exponential backoff factor
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio
            
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logging.error(f"{func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logging.warning(f"{func.__name__} attempt {attempt + 1} failed: {str(e)}. Retrying in {current_delay:.2f} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
            
            return None
        
        return wrapper
    return decorator
