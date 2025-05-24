"""
MACD Histogram Strategy
Uses MACD histogram patterns and divergences for signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class MACDHistogramStrategy(BaseStrategy):
    """
    MACD Histogram strategy that analyzes MACD line, signal line, and histogram patterns
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_period': 12,            # Fast EMA period
            'slow_period': 26,            # Slow EMA period
            'signal_period': 9,           # Signal line EMA period
            'histogram_threshold': 0.001,  # Minimum histogram value for signals
            'divergence_lookback': 20,    # Periods to look back for divergences
            'zero_line_cross_confirm': True,  # Require zero line cross confirmation
            'histogram_momentum_periods': 5,  # Periods for histogram momentum
            'trend_filter_period': 50,    # Trend filter MA period
            'volume_confirmation': True,   # Require volume confirmation
            'min_volume_ratio': 1.2,     # Minimum volume ratio
            'divergence_min_peaks': 2,    # Minimum peaks for divergence
            'histogram_acceleration_threshold': 0.0005,  # Histogram acceleration threshold
            'macd_signal_cross_confirm': True,  # Require MACD-Signal crossover
            'stop_loss_atr_mult': 2.0,
            'take_profit_atr_mult': 3.5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MACD Histogram", default_params)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate MACD Histogram signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['slow_period'] + 50:
                continue
                
            try:
                # Calculate MACD indicators
                df_indicators = self._calculate_macd_indicators(df.copy())
                
                # Generate signals based on MACD histogram analysis
                symbol_signals = self._generate_macd_signals(symbol, df_indicators)
                signals.extend(symbol_signals)
                
            except Exception as e:
                print(f"Error generating MACD Histogram signals for {symbol}: {e}")
                continue
                
        return signals
    
    def _calculate_macd_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD, Signal line, Histogram, and related indicators
        """
        # MACD calculation
        ema_fast = df['Close'].ewm(span=self.parameters['fast_period']).mean()
        ema_slow = df['Close'].ewm(span=self.parameters['slow_period']).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=self.parameters['signal_period']).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Histogram analysis
        df['Histogram_Momentum'] = df['MACD_Histogram'].diff(self.parameters['histogram_momentum_periods'])
        df['Histogram_Acceleration'] = df['Histogram_Momentum'].diff()
        
        # Zero line crosses
        df['MACD_Above_Zero'] = df['MACD'] > 0
        df['MACD_Zero_Cross_Up'] = (df['MACD'] > 0) & (df['MACD'].shift(1) <= 0)
        df['MACD_Zero_Cross_Down'] = (df['MACD'] < 0) & (df['MACD'].shift(1) >= 0)
        
        # Signal line crosses
        df['MACD_Above_Signal'] = df['MACD'] > df['MACD_Signal']
        df['MACD_Signal_Cross_Up'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        df['MACD_Signal_Cross_Down'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        
        # Histogram patterns
        df['Histogram_Positive'] = df['MACD_Histogram'] > 0
        df['Histogram_Increasing'] = df['MACD_Histogram'] > df['MACD_Histogram'].shift(1)
        df['Histogram_Peak'] = self._identify_histogram_peaks(df['MACD_Histogram'])
        df['Histogram_Trough'] = self._identify_histogram_troughs(df['MACD_Histogram'])
        
        # Divergences
        df['Price_Peaks'] = self._identify_price_peaks(df['Close'])
        df['Price_Troughs'] = self._identify_price_troughs(df['Close'])
        
        # Trend filter
        trend_period = self.parameters['trend_filter_period']
        df['Trend_MA'] = df['Close'].rolling(window=trend_period).mean()
        df['Trend_Direction'] = np.where(df['Close'] > df['Trend_MA'], 1, -1)
        
        # Volume analysis
        if self.parameters['volume_confirmation']:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # MACD strength
        df['MACD_Strength'] = abs(df['MACD']) / df['Close'].rolling(window=20).std()
        
        # Histogram compression/expansion
        df['Histogram_Volatility'] = df['MACD_Histogram'].rolling(window=10).std()
        
        return df
    
    def _identify_histogram_peaks(self, histogram: pd.Series) -> pd.Series:
        """
        Identify peaks in MACD histogram
        """
        peaks = pd.Series(False, index=histogram.index)
        
        for i in range(2, len(histogram) - 2):
            if (histogram.iloc[i] > histogram.iloc[i-1] and 
                histogram.iloc[i] > histogram.iloc[i+1] and
                histogram.iloc[i] > histogram.iloc[i-2] and 
                histogram.iloc[i] > histogram.iloc[i+2]):
                peaks.iloc[i] = True
        
        return peaks
    
    def _identify_histogram_troughs(self, histogram: pd.Series) -> pd.Series:
        """
        Identify troughs in MACD histogram
        """
        troughs = pd.Series(False, index=histogram.index)
        
        for i in range(2, len(histogram) - 2):
            if (histogram.iloc[i] < histogram.iloc[i-1] and 
                histogram.iloc[i] < histogram.iloc[i+1] and
                histogram.iloc[i] < histogram.iloc[i-2] and 
                histogram.iloc[i] < histogram.iloc[i+2]):
                troughs.iloc[i] = True
        
        return troughs
    
    def _identify_price_peaks(self, prices: pd.Series) -> pd.Series:
        """
        Identify peaks in price
        """
        peaks = pd.Series(False, index=prices.index)
        
        for i in range(2, len(prices) - 2):
            if (prices.iloc[i] > prices.iloc[i-1] and 
                prices.iloc[i] > prices.iloc[i+1] and
                prices.iloc[i] > prices.iloc[i-2] and 
                prices.iloc[i] > prices.iloc[i+2]):
                peaks.iloc[i] = True
        
        return peaks
    
    def _identify_price_troughs(self, prices: pd.Series) -> pd.Series:
        """
        Identify troughs in price
        """
        troughs = pd.Series(False, index=prices.index)
        
        for i in range(2, len(prices) - 2):
            if (prices.iloc[i] < prices.iloc[i-1] and 
                prices.iloc[i] < prices.iloc[i+1] and
                prices.iloc[i] < prices.iloc[i-2] and 
                prices.iloc[i] < prices.iloc[i+2]):
                troughs.iloc[i] = True
        
        return troughs
    
    def _generate_macd_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on MACD histogram analysis
        """
        signals = []
        
        for i in range(len(df) - 10, len(df)):
            if i < 50:  # Need enough data
                continue
                
            current = df.iloc[i]
            recent_df = df.iloc[max(0, i-20):i+1]
            
            signal = self._analyze_macd_patterns(symbol, current, recent_df, i, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_macd_patterns(self, symbol: str, current: pd.Series, 
                             recent_df: pd.DataFrame, index: int, 
                             full_df: pd.DataFrame) -> Signal:
        """
        Analyze MACD histogram patterns for signal generation
        """
        signal_strength = 0
        signal_type = None
        reasons = []
        
        # Check basic requirements
        if (abs(current['MACD_Histogram']) < self.parameters['histogram_threshold'] or
            pd.isna(current['MACD']) or pd.isna(current['MACD_Signal'])):
            return None
        
        # Volume confirmation
        if self.parameters['volume_confirmation']:
            if current['Volume_Ratio'] < self.parameters['min_volume_ratio']:
                return None
        
        # 1. Histogram momentum change
        histogram_momentum = current['Histogram_Momentum']
        if abs(histogram_momentum) > 0.0001:
            if histogram_momentum > 0:
                signal_type = SignalType.BUY
                signal_strength += 0.25
                reasons.append("histogram_momentum_up")
            else:
                signal_type = SignalType.SELL
                signal_strength += 0.25
                reasons.append("histogram_momentum_down")
        
        # 2. Histogram acceleration
        histogram_acceleration = current['Histogram_Acceleration']
        if not pd.isna(histogram_acceleration):
            if abs(histogram_acceleration) > self.parameters['histogram_acceleration_threshold']:
                if (histogram_acceleration > 0 and signal_type == SignalType.BUY):
                    signal_strength += 0.2
                    reasons.append("histogram_acceleration_up")
                elif (histogram_acceleration < 0 and signal_type == SignalType.SELL):
                    signal_strength += 0.2
                    reasons.append("histogram_acceleration_down")
        
        # 3. MACD-Signal line crossover
        if self.parameters['macd_signal_cross_confirm']:
            if current['MACD_Signal_Cross_Up'] and signal_type == SignalType.BUY:
                signal_strength += 0.3
                reasons.append("macd_signal_cross_up")
            elif current['MACD_Signal_Cross_Down'] and signal_type == SignalType.SELL:
                signal_strength += 0.3
                reasons.append("macd_signal_cross_down")
        
        # 4. Zero line cross
        if self.parameters['zero_line_cross_confirm']:
            if current['MACD_Zero_Cross_Up'] and signal_type == SignalType.BUY:
                signal_strength += 0.25
                reasons.append("macd_zero_cross_up")
            elif current['MACD_Zero_Cross_Down'] and signal_type == SignalType.SELL:
                signal_strength += 0.25
                reasons.append("macd_zero_cross_down")
        
        # 5. Histogram pattern analysis
        if current['Histogram_Peak'] and signal_type == SignalType.SELL:
            signal_strength += 0.15
            reasons.append("histogram_peak")
        elif current['Histogram_Trough'] and signal_type == SignalType.BUY:
            signal_strength += 0.15
            reasons.append("histogram_trough")
        
        # 6. Divergence analysis
        divergence_signal = self._check_macd_divergence(full_df, index)
        if divergence_signal:
            if divergence_signal == 'bullish' and signal_type == SignalType.BUY:
                signal_strength += 0.3
                reasons.append("bullish_divergence")
            elif divergence_signal == 'bearish' and signal_type == SignalType.SELL:
                signal_strength += 0.3
                reasons.append("bearish_divergence")
        
        # 7. Trend alignment
        trend_direction = current['Trend_Direction']
        if ((signal_type == SignalType.BUY and trend_direction > 0) or
            (signal_type == SignalType.SELL and trend_direction < 0)):
            signal_strength += 0.1
            reasons.append("trend_aligned")
        
        # 8. MACD strength
        macd_strength = current['MACD_Strength']
        if not pd.isna(macd_strength) and macd_strength > 1:
            signal_strength += 0.1
            reasons.append("strong_macd")
        
        # 9. Histogram position relative to zero
        histogram = current['MACD_Histogram']
        if signal_type == SignalType.BUY and histogram < 0:
            signal_strength += 0.1
            reasons.append("histogram_below_zero_buy")
        elif signal_type == SignalType.SELL and histogram > 0:
            signal_strength += 0.1
            reasons.append("histogram_above_zero_sell")
        
        # Only generate signal if strength is sufficient
        if signal_strength < 0.5 or signal_type is None:
            return None
        
        # Cap signal strength
        signal_strength = min(signal_strength, 1.0)
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=signal_strength,
            price=current['Close'],
            timestamp=current.name,
            metadata={
                'strategy': 'macd_histogram',
                'reasons': reasons,
                'macd': current['MACD'],
                'macd_signal': current['MACD_Signal'],
                'histogram': histogram,
                'histogram_momentum': histogram_momentum,
                'histogram_acceleration': histogram_acceleration,
                'macd_strength': macd_strength,
                'trend_direction': trend_direction,
                'volume_ratio': current.get('Volume_Ratio', 1.0)
            }
        )
    
    def _check_macd_divergence(self, df: pd.DataFrame, current_index: int) -> str:
        """
        Check for MACD-Price divergences
        """
        lookback = self.parameters['divergence_lookback']
        min_peaks = self.parameters['divergence_min_peaks']
        
        if current_index < lookback:
            return None
        
        recent_data = df.iloc[current_index - lookback:current_index + 1]
        
        # Find recent peaks and troughs
        price_peaks = recent_data[recent_data['Price_Peaks']].index
        price_troughs = recent_data[recent_data['Price_Troughs']].index
        histogram_peaks = recent_data[recent_data['Histogram_Peak']].index
        histogram_troughs = recent_data[recent_data['Histogram_Trough']].index
        
        # Check for bullish divergence (price makes lower lows, MACD makes higher lows)
        if len(price_troughs) >= min_peaks and len(histogram_troughs) >= min_peaks:
            if len(price_troughs) >= 2 and len(histogram_troughs) >= 2:
                recent_price_troughs = price_troughs[-2:]
                recent_histogram_troughs = histogram_troughs[-2:]
                
                if len(recent_price_troughs) == 2 and len(recent_histogram_troughs) == 2:
                    price_low1 = df.loc[recent_price_troughs[0], 'Close']
                    price_low2 = df.loc[recent_price_troughs[1], 'Close']
                    hist_low1 = df.loc[recent_histogram_troughs[0], 'MACD_Histogram']
                    hist_low2 = df.loc[recent_histogram_troughs[1], 'MACD_Histogram']
                    
                    if price_low2 < price_low1 and hist_low2 > hist_low1:
                        return 'bullish'
        
        # Check for bearish divergence (price makes higher highs, MACD makes lower highs)
        if len(price_peaks) >= min_peaks and len(histogram_peaks) >= min_peaks:
            if len(price_peaks) >= 2 and len(histogram_peaks) >= 2:
                recent_price_peaks = price_peaks[-2:]
                recent_histogram_peaks = histogram_peaks[-2:]
                
                if len(recent_price_peaks) == 2 and len(recent_histogram_peaks) == 2:
                    price_high1 = df.loc[recent_price_peaks[0], 'Close']
                    price_high2 = df.loc[recent_price_peaks[1], 'Close']
                    hist_high1 = df.loc[recent_histogram_peaks[0], 'MACD_Histogram']
                    hist_high2 = df.loc[recent_histogram_peaks[1], 'MACD_Histogram']
                    
                    if price_high2 > price_high1 and hist_high2 < hist_high1:
                        return 'bearish'
        
        return None
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for MACD histogram trades
        """
        # Base position size
        base_size = 0.10  # 10% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on MACD strength
        macd_strength = signal.metadata.get('macd_strength', 1)
        macd_multiplier = 1 + min(macd_strength / 2, 0.3)  # Up to 30% increase
        
        # Adjust based on volume (if available)
        volume_ratio = signal.metadata.get('volume_ratio', 1)
        volume_multiplier = min(volume_ratio / self.parameters['min_volume_ratio'], 1.5)
        
        # Bonus for divergence signals
        reasons = signal.metadata.get('reasons', [])
        divergence_bonus = 1.2 if any('divergence' in reason for reason in reasons) else 1.0
        
        # Calculate final position size
        position_size = (base_size * strength_multiplier * macd_multiplier * 
                        volume_multiplier * divergence_bonus)
        
        # Cap at maximum position size
        max_position = 0.20  # Maximum 20% for MACD histogram trades
        position_size = min(position_size, max_position)
        
        return position_size * portfolio_value / signal.price

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate a single signal for demo trading (adapter for generate_signals)
        """
        try:
            # Convert single symbol data to multi-symbol format expected by generate_signals
            temp_data = {'TEMP_SYMBOL': data}
            signals = self.generate_signals(temp_data)
            
            if signals:
                signal = signals[0]  # Take the first signal
                return {
                    'action': signal.signal_type.name,
                    'strength': signal.strength,
                    'price': signal.price,
                    'metadata': signal.metadata
                }
            
            return {'action': 'HOLD', 'strength': 0, 'price': data.iloc[-1]['Close'], 'metadata': {}}
            
        except Exception as e:
            print(f"Error in MACD Histogram generate_signal: {e}")
            return {'action': 'HOLD', 'strength': 0, 'price': data.iloc[-1]['Close'], 'metadata': {}}
