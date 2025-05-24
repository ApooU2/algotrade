"""
Breakout Strategy
Identifies and trades price breakouts from consolidation patterns
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal, SignalType

class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy that identifies price breakouts from support/resistance levels
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 20,
            'volume_confirmation': True,
            'min_volume_ratio': 1.5,
            'breakout_threshold': 0.01,  # 1% breakout threshold
            'consolidation_period': 10,
            'atr_period': 14,
            'atr_multiplier': 2,
            'min_consolidation_days': 5,
            'max_consolidation_days': 50,
            'stop_loss_atr_mult': 1.5,
            'take_profit_ratio': 3  # Risk:Reward ratio
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Breakout", default_params)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate breakout trading signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['lookback_period']:
                continue
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Identify consolidation periods and breakouts
            df = self._identify_consolidation(df)
            
            # Get latest data
            latest = df.iloc[-1]
            recent = df.tail(self.parameters['consolidation_period'])
            
            # Check breakout conditions
            signal = self._check_breakout_conditions(symbol, latest, recent, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate breakout-related indicators
        """
        lookback = self.parameters['lookback_period']
        
        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=lookback).min()
        df['Resistance'] = df['High'].rolling(window=lookback).max()
        
        # Price range and volatility
        df['Price_range'] = df['High'] - df['Low']
        df['ATR'] = self._calculate_atr(df, self.parameters['atr_period'])
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price position within recent range
        df['Price_position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
        
        # Bollinger Bands for squeeze detection
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Moving averages for trend confirmation
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Price momentum
        df['Momentum'] = df['Close'].pct_change(5)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _identify_consolidation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify consolidation periods
        """
        # Calculate price volatility over rolling windows
        df['Price_volatility'] = df['Close'].rolling(window=10).std() / df['Close'].rolling(window=10).mean()
        
        # Consolidation indicator (low volatility periods)
        volatility_threshold = df['Price_volatility'].rolling(window=50).quantile(0.3)
        df['In_consolidation'] = df['Price_volatility'] < volatility_threshold
        
        # Bollinger Band squeeze (another consolidation indicator)
        bb_width_threshold = df['BB_width'].rolling(window=50).quantile(0.2)
        df['BB_squeeze'] = df['BB_width'] < bb_width_threshold
        
        # Combined consolidation signal
        df['Consolidation_signal'] = df['In_consolidation'] | df['BB_squeeze']
        
        # Count consecutive consolidation days
        df['Consolidation_days'] = df['Consolidation_signal'].groupby(
            (df['Consolidation_signal'] != df['Consolidation_signal'].shift()).cumsum()
        ).cumsum()
        
        df['Consolidation_days'] = df['Consolidation_days'].where(df['Consolidation_signal'], 0)
        
        return df
    
    def _check_breakout_conditions(self, symbol: str, latest: pd.Series, 
                                 recent: pd.DataFrame, df: pd.DataFrame) -> Signal:
        """
        Check for breakout conditions
        """
        # Check if we're coming out of consolidation
        consolidation_days = latest['Consolidation_days']
        was_consolidating = recent['Consolidation_signal'].any()
        
        if not was_consolidating or consolidation_days > self.parameters['max_consolidation_days']:
            return None
        
        # Sufficient consolidation period
        if consolidation_days < self.parameters['min_consolidation_days']:
            return None
        
        # Get support and resistance levels
        support = latest['Support']
        resistance = latest['Resistance']
        current_price = latest['Close']
        high = latest['High']
        low = latest['Low']
        
        # Calculate breakout thresholds
        resistance_breakout = resistance * (1 + self.parameters['breakout_threshold'])
        support_breakout = support * (1 - self.parameters['breakout_threshold'])
        
        # Volume confirmation
        volume_confirmed = True
        if self.parameters['volume_confirmation']:
            volume_confirmed = latest['Volume_ratio'] >= self.parameters['min_volume_ratio']
        
        # Upward breakout conditions
        upward_breakout_conditions = [
            high > resistance_breakout,  # Price broke above resistance
            current_price > resistance,  # Close above resistance
            volume_confirmed,  # Volume confirmation
            latest['SMA_20'] > latest['SMA_50'],  # Trend confirmation
            latest['Momentum'] > 0,  # Positive momentum
            latest['ATR'] / current_price > 0.01  # Sufficient volatility
        ]
        
        # Downward breakout conditions
        downward_breakout_conditions = [
            low < support_breakout,  # Price broke below support
            current_price < support,  # Close below support
            volume_confirmed,  # Volume confirmation
            latest['SMA_20'] < latest['SMA_50'],  # Trend confirmation
            latest['Momentum'] < 0,  # Negative momentum
            latest['ATR'] / current_price > 0.01  # Sufficient volatility
        ]
        
        # Calculate signal strength
        upward_strength = sum(upward_breakout_conditions) / len(upward_breakout_conditions)
        downward_strength = sum(downward_breakout_conditions) / len(downward_breakout_conditions)
        
        # Additional strength factors
        consolidation_factor = min(consolidation_days / self.parameters['max_consolidation_days'], 1.0)
        volume_factor = min(latest['Volume_ratio'] / self.parameters['min_volume_ratio'], 2.0)
        range_factor = (resistance - support) / current_price  # Wider range = stronger breakout
        
        if upward_strength >= 0.8:  # Strong upward breakout
            strength = upward_strength * (1 + consolidation_factor * 0.3) * (1 + volume_factor * 0.2) * (1 + range_factor * 2)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=min(strength, 1.0),
                price=current_price,
                timestamp=latest.name,
                metadata={
                    'strategy': 'breakout',
                    'breakout_type': 'upward',
                    'resistance_level': resistance,
                    'support_level': support,
                    'consolidation_days': consolidation_days,
                    'volume_ratio': latest['Volume_ratio'],
                    'atr': latest['ATR'],
                    'range_factor': range_factor
                }
            )
        
        elif downward_strength >= 0.8:  # Strong downward breakout
            strength = downward_strength * (1 + consolidation_factor * 0.3) * (1 + volume_factor * 0.2) * (1 + range_factor * 2)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=min(strength, 1.0),
                price=current_price,
                timestamp=latest.name,
                metadata={
                    'strategy': 'breakout',
                    'breakout_type': 'downward',
                    'resistance_level': resistance,
                    'support_level': support,
                    'consolidation_days': consolidation_days,
                    'volume_ratio': latest['Volume_ratio'],
                    'atr': latest['ATR'],
                    'range_factor': range_factor
                }
            )
        
        return None
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for breakout trades
        """
        # Base position size (breakouts can be aggressive)
        base_size = 0.10  # 10% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on consolidation period (longer consolidation = bigger move expected)
        consolidation_days = signal.metadata.get('consolidation_days', 0)
        consolidation_multiplier = 1 + min(consolidation_days / 30, 0.5)  # Up to 50% increase
        
        # Adjust based on range factor (wider consolidation range = bigger position)
        range_factor = signal.metadata.get('range_factor', 0)
        range_multiplier = 1 + min(range_factor * 10, 0.4)  # Up to 40% increase
        
        # Volume factor
        volume_ratio = signal.metadata.get('volume_ratio', 1)
        volume_multiplier = min(volume_ratio / self.parameters['min_volume_ratio'], 1.5)
        
        # Calculate final position size
        position_size = (base_size * strength_multiplier * consolidation_multiplier * 
                        range_multiplier * volume_multiplier)
        
        # Cap at maximum position size
        max_position = 0.25  # Maximum 25% for breakout trades
        position_size = min(position_size, max_position)
        
        return position_size * portfolio_value / signal.price
