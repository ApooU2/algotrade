"""
RSI Divergence Strategy
Identifies bullish and bearish divergences between price and RSI
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal, SignalType

class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI Divergence Strategy that identifies divergences between price and RSI
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'lookback_period': 20,
            'min_rsi_oversold': 35,
            'max_rsi_overbought': 65,
            'min_price_change': 0.02,  # 2% minimum price change for divergence
            'divergence_periods': 5,   # Number of periods to confirm divergence
            'volume_confirmation': True,
            'min_volume_ratio': 1.2
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("RSI Divergence", default_params)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate RSI divergence trading signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['lookback_period']:
                continue
            
            # Calculate RSI
            df = self._calculate_rsi(df)
            
            # Find divergences
            divergence_signal = self._detect_divergence(symbol, df)
            
            if divergence_signal:
                signals.append(divergence_signal)
        
        return signals
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        period = self.parameters['rsi_period']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _detect_divergence(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Detect bullish and bearish divergences
        """
        lookback = self.parameters['lookback_period']
        recent_data = df.tail(lookback)
        
        if len(recent_data) < lookback:
            return None
        
        # Get price and RSI data
        prices = recent_data['Close'].values
        rsi_values = recent_data['RSI'].values
        
        # Find peaks and troughs
        price_peaks, price_troughs = self._find_peaks_troughs(prices)
        rsi_peaks, rsi_troughs = self._find_peaks_troughs(rsi_values)
        
        # Check for bullish divergence (price makes lower low, RSI makes higher low)
        bullish_div = self._check_bullish_divergence(price_troughs, rsi_troughs, prices, rsi_values)
        
        # Check for bearish divergence (price makes higher high, RSI makes lower high)
        bearish_div = self._check_bearish_divergence(price_peaks, rsi_peaks, prices, rsi_values)
        
        current_rsi = rsi_values[-1]
        current_price = prices[-1]
        
        # Generate signals
        if bullish_div and current_rsi < self.parameters['min_rsi_oversold']:
            # Check volume confirmation if required
            if self._check_volume_confirmation(recent_data):
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=self._calculate_signal_strength(current_rsi, 'bullish'),
                    price=current_price,
                    timestamp=recent_data.index[-1],
                    metadata={
                        'strategy': 'rsi_divergence',
                        'divergence_type': 'bullish',
                        'current_rsi': current_rsi,
                        'signal_reason': 'Bullish RSI divergence detected'
                    }
                )
        
        elif bearish_div and current_rsi > self.parameters['max_rsi_overbought']:
            if self._check_volume_confirmation(recent_data):
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=self._calculate_signal_strength(current_rsi, 'bearish'),
                    price=current_price,
                    timestamp=recent_data.index[-1],
                    metadata={
                        'strategy': 'rsi_divergence',
                        'divergence_type': 'bearish',
                        'current_rsi': current_rsi,
                        'signal_reason': 'Bearish RSI divergence detected'
                    }
                )
        
        return None
    
    def _find_peaks_troughs(self, data: np.array) -> tuple:
        """Find peaks and troughs in data"""
        peaks = []
        troughs = []
        
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
            elif data[i] < data[i-1] and data[i] < data[i+1]:
                troughs.append(i)
        
        return peaks, troughs
    
    def _check_bullish_divergence(self, price_troughs: List, rsi_troughs: List, 
                                 prices: np.array, rsi_values: np.array) -> bool:
        """Check for bullish divergence pattern"""
        if len(price_troughs) < 2 or len(rsi_troughs) < 2:
            return False
        
        # Get last two troughs
        last_price_trough = price_troughs[-1]
        prev_price_trough = price_troughs[-2]
        last_rsi_trough = rsi_troughs[-1]
        prev_rsi_trough = rsi_troughs[-2]
        
        # Check if price made lower low but RSI made higher low
        price_lower_low = prices[last_price_trough] < prices[prev_price_trough]
        rsi_higher_low = rsi_values[last_rsi_trough] > rsi_values[prev_rsi_trough]
        
        # Ensure minimum price change
        price_change = abs(prices[last_price_trough] - prices[prev_price_trough]) / prices[prev_price_trough]
        
        return price_lower_low and rsi_higher_low and price_change >= self.parameters['min_price_change']
    
    def _check_bearish_divergence(self, price_peaks: List, rsi_peaks: List,
                                 prices: np.array, rsi_values: np.array) -> bool:
        """Check for bearish divergence pattern"""
        if len(price_peaks) < 2 or len(rsi_peaks) < 2:
            return False
        
        # Get last two peaks
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        last_rsi_peak = rsi_peaks[-1]
        prev_rsi_peak = rsi_peaks[-2]
        
        # Check if price made higher high but RSI made lower high
        price_higher_high = prices[last_price_peak] > prices[prev_price_peak]
        rsi_lower_high = rsi_values[last_rsi_peak] < rsi_values[prev_rsi_peak]
        
        # Ensure minimum price change
        price_change = abs(prices[last_price_peak] - prices[prev_price_peak]) / prices[prev_price_peak]
        
        return price_higher_high and rsi_lower_high and price_change >= self.parameters['min_price_change']
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if volume confirms the signal"""
        if not self.parameters['volume_confirmation']:
            return True
        
        recent_volume = df['Volume'].tail(3).mean()
        avg_volume = df['Volume'].tail(20).mean()
        
        return recent_volume >= avg_volume * self.parameters['min_volume_ratio']
    
    def _calculate_signal_strength(self, current_rsi: float, divergence_type: str) -> float:
        """Calculate signal strength based on RSI level and divergence quality"""
        if divergence_type == 'bullish':
            # Stronger signal when RSI is more oversold
            base_strength = max(0, (40 - current_rsi) / 40)
        else:  # bearish
            # Stronger signal when RSI is more overbought
            base_strength = max(0, (current_rsi - 60) / 40)
        
        return min(0.9, max(0.5, base_strength))
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for RSI divergence trades
        """
        # Base position size
        base_size = 0.06  # 6% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on RSI extremity
        current_rsi = signal.metadata.get('current_rsi', 50)
        if signal.signal_type == SignalType.BUY:
            rsi_multiplier = max(1.0, (35 - current_rsi) / 20)  # More oversold = larger position
        else:
            rsi_multiplier = max(1.0, (current_rsi - 65) / 20)  # More overbought = larger position
        
        # Volatility adjustment
        vol_adjustment = max(0.5, min(1.5, 0.2 / volatility))
        
        # Calculate final position size
        position_size = base_size * strength_multiplier * rsi_multiplier * vol_adjustment
        
        # Cap at maximum position size
        max_position = 0.20  # Maximum 20% for divergence trades
        position_size = min(position_size, max_position)
        
        return position_size * portfolio_value / signal.price
