"""
Base strategy class for all trading strategies
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class Signal:
    symbol: str
    signal_type: SignalType
    strength: float  # Signal strength 0-1
    price: float
    timestamp: pd.Timestamp
    metadata: Dict = None

class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    """
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
        self.signals = []
        self.positions = {}
        
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals based on the data
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size based on signal and risk parameters
        """
        pass
    
    def validate_signal(self, signal: Signal, current_positions: Dict) -> bool:
        """
        Validate if a signal should be executed
        """
        # Basic validation logic
        if signal.strength < 0.5:  # Minimum signal strength threshold
            return False
        
        # Check if we already have a position in the opposite direction
        current_position = current_positions.get(signal.symbol, 0)
        if (signal.signal_type == SignalType.BUY and current_position < 0) or \
           (signal.signal_type == SignalType.SELL and current_position > 0):
            return True
        
        # Don't add to existing position unless signal is very strong
        if current_position != 0 and signal.strength < 0.8:
            return False
        
        return True
    
    def get_stop_loss_price(self, entry_price: float, signal_type: SignalType, 
                           atr: float = None) -> float:
        """
        Calculate stop loss price
        """
        stop_loss_pct = self.parameters.get('stop_loss_pct', 0.05)
        
        if atr:
            # Use ATR-based stop loss
            stop_distance = atr * self.parameters.get('atr_multiplier', 2)
        else:
            # Use percentage-based stop loss
            stop_distance = entry_price * stop_loss_pct
        
        if signal_type == SignalType.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def get_take_profit_price(self, entry_price: float, signal_type: SignalType, 
                             atr: float = None) -> float:
        """
        Calculate take profit price
        """
        take_profit_pct = self.parameters.get('take_profit_pct', 0.15)
        
        if atr:
            # Use ATR-based take profit
            profit_distance = atr * self.parameters.get('atr_profit_multiplier', 3)
        else:
            # Use percentage-based take profit
            profit_distance = entry_price * take_profit_pct
        
        if signal_type == SignalType.BUY:
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio of the strategy
        """
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
    
    def get_strategy_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive strategy metrics
        """
        return {
            'total_return': returns.sum(),
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown((1 + returns).cumprod()),
            'win_rate': (returns > 0).mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
