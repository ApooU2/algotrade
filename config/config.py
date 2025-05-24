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
                'mean_reversion',
                'momentum',
                'breakout',
                'ml_ensemble',
                'pairs_trading'
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
