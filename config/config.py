"""
Configuration settings for the algorithmic trading bot
"""
import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    # API Configuration
    ALPACA_API_KEY: str = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY: str = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL: str = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')  # Paper trading by default
    
    # Trading Parameters
    INITIAL_CAPITAL: float = 100000.0  # Starting capital
    MAX_POSITION_SIZE: float = 0.1  # Maximum 10% of portfolio per position
    MAX_DAILY_LOSS: float = 0.02  # Maximum 2% daily loss
    STOP_LOSS_PCT: float = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT: float = 0.15  # 15% take profit
    
    # Strategy Configuration
    STRATEGIES: List[str] = None
    SYMBOLS: List[str] = None
    TIMEFRAME: str = '1D'  # Daily timeframe
    
    # Risk Management
    VAR_CONFIDENCE: float = 0.05  # 95% VaR
    MAX_CORRELATION: float = 0.7  # Maximum correlation between positions
    
    # Backtesting
    BACKTEST_START: str = '2020-01-01'
    BACKTEST_END: str = '2024-12-31'
    COMMISSION: float = 0.001  # 0.1% commission
    
    def __post_init__(self):
        if self.STRATEGIES is None:
            self.STRATEGIES = [
                # Core Technical Strategies
                'mean_reversion',
                'momentum', 
                'breakout',
                # Advanced Technical Strategies
                'rsi_divergence',
                'vwap',
                'bollinger_squeeze',
                'macd_histogram',
                'ichimoku',
                'support_resistance',
                # Volume-Based Strategies
                'volume_profile',
                # Market Microstructure
                'market_microstructure',
                # Gap Trading
                'gap_trading',
                # Statistical Arbitrage
                'pairs_trading',
                # Machine Learning
                'ml_ensemble'
            ]
        
        if self.SYMBOLS is None:
            self.SYMBOLS = [
                'SPY', 'QQQ', 'IWM',  # ETFs
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech stocks
                'JPM', 'BAC', 'WFC',  # Financials
                'XOM', 'CVX',  # Energy
                'JNJ', 'PFE',  # Healthcare
            ]

# Global configuration instance
CONFIG = TradingConfig()

# Strategy-specific configurations
MEAN_REVERSION_CONFIG = {
    'bollinger_period': 20,
    'bollinger_std': 2,
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'volume_threshold': 1.5,
    'lookback_period': 50
}

MOMENTUM_CONFIG = {
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'trend_period': 50,
    'momentum_period': 10,
    'volume_confirmation': True,
    'min_trend_strength': 0.02
}

BREAKOUT_CONFIG = {
    'consolidation_period': 20,
    'breakout_threshold': 0.02,
    'volume_multiplier': 2.0,
    'atr_period': 14,
    'atr_multiplier': 1.5,
    'min_consolidation_days': 5
}

ML_CONFIG = {
    'models': ['random_forest', 'gradient_boosted_trees'],
    'lookback_period': 60,
    'prediction_horizon': 5,
    'min_prediction_confidence': 0.6,
    'retrain_frequency': 30,
    'feature_importance_threshold': 0.01,
    'ensemble_threshold': 0.7
}

PAIRS_TRADING_CONFIG = {
    'lookback_period': 252,
    'z_score_entry': 2.0,
    'z_score_exit': 0.5,
    'correlation_threshold': 0.8,
    'cointegration_threshold': 0.05,
    'max_holding_period': 10
}

# Time-based Trading Configuration
TIME_BASED_CONFIG = {
    'max_holding_period_days': 7,  # Maximum holding period for any position
    'market_hours_only_stocks': True,  # Only trade stocks during market hours
    'allow_crypto_24_7': True,  # Allow crypto trading 24/7
    'force_close_before_weekend': False,  # Force close positions before weekend
    'max_aftermarket_exposure': 0.1,  # Maximum portfolio exposure during aftermarket hours
    'position_age_check_interval': 3600,  # Check position ages every hour (seconds)
}

# Advanced Technical Strategy Configurations
RSI_DIVERGENCE_CONFIG = {
    'rsi_period': 14,
    'lookback_period': 50,
    'divergence_threshold': 0.7,
    'min_divergence_bars': 5,
    'price_change_threshold': 0.02,
    'volume_confirmation': True
}

VWAP_CONFIG = {
    'vwap_period': 20,
    'band_std': 1.5,
    'volume_threshold': 1.2,
    'trend_confirmation': True,
    'multi_timeframe': True,
    'anchored_vwap': True
}

BOLLINGER_SQUEEZE_CONFIG = {
    'bb_period': 20,
    'bb_std': 2.0,
    'kc_period': 20,
    'kc_atr_mult': 1.5,
    'squeeze_threshold': 0.9,
    'momentum_period': 12,
    'min_squeeze_bars': 5
}

MACD_HISTOGRAM_CONFIG = {
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'histogram_threshold': 0.02,
    'divergence_detection': True,
    'zero_line_analysis': True
}

ICHIMOKU_CONFIG = {
    'tenkan_period': 9,
    'kijun_period': 26,
    'senkou_span_b': 52,
    'displacement': 26,
    'cloud_analysis': True,
    'trend_strength_threshold': 0.7
}

SUPPORT_RESISTANCE_CONFIG = {
    'lookback_period': 50,
    'level_strength_threshold': 3,
    'fibonacci_levels': True,
    'pivot_points': True,
    'min_level_age': 5,
    'max_level_age': 200
}

# Volume-Based Strategy Configurations
VOLUME_PROFILE_CONFIG = {
    'value_area_percent': 70,
    'profile_period': 20,
    'price_bins': 50,
    'high_volume_node_threshold': 1.5,
    'poc_bounce_threshold': 0.005,
    'va_breakout_threshold': 0.01
}

# Market Microstructure Configuration
MARKET_MICROSTRUCTURE_CONFIG = {
    'spread_threshold': 0.005,
    'volume_imbalance_threshold': 0.6,
    'aggressive_order_threshold': 0.7,
    'liquidity_threshold': 100000,
    'price_efficiency_threshold': 0.8,
    'order_flow_window': 15
}

# Gap Trading Configuration
GAP_TRADING_CONFIG = {
    'min_gap_percent': 1.0,
    'max_gap_percent': 8.0,
    'gap_fill_threshold': 0.75,
    'continuation_threshold': 0.25,
    'volume_confirmation': True,
    'min_volume_ratio': 1.5
}

RISK_CONFIG = {
    'max_portfolio_risk': 0.02,
    'var_confidence': 0.05,
    'max_position_size': 0.1,
    'max_correlation': 0.7,
    'stop_loss_pct': 0.05,
    'position_sizing_method': 'kelly',
    'risk_free_rate': 0.02
}

BACKTESTING_CONFIG = {
    'initial_capital': 100000,
    'commission': 0.001,
    'slippage': 0.0005,
    'benchmark': 'SPY',
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'rebalance_frequency': 'daily'
}

TRADING_CONFIG = CONFIG.__dict__
