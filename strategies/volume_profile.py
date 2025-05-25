"""
Volume Profile Strategy - Identifies high-volume nodes and value areas for trading
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class VolumeProfileStrategy(BaseStrategy):
    """
    Volume Profile strategy that identifies value areas, high-volume nodes,
    and volume imbalances for trading opportunities
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'value_area_percent': 70,      # Value area percentage (70% of volume)
            'profile_period': 20,          # Period for building volume profile
            'min_volume_threshold': 50000, # Minimum volume threshold
            'poc_bounce_threshold': 0.005, # POC bounce threshold (0.5%)
            'va_breakout_threshold': 0.01, # Value area breakout threshold (1%)
            'volume_imbalance_ratio': 2.0, # Volume imbalance detection ratio
            'price_bins': 50,              # Number of price bins for profile
            'min_time_at_level': 3,        # Minimum time spent at price level
            'high_volume_node_threshold': 1.5, # High volume node threshold
            'stop_loss_atr_mult': 2.0,
            'take_profit_ratio': 2.5,
            'lookback_period': 100
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Volume Profile", default_params)
        
        # Store volume profile data
        self.volume_profiles = {}
        self.value_areas = {}
        self.poc_levels = {}  # Point of Control levels
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate Volume Profile signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['lookback_period']:
                continue
                
            try:
                # Calculate indicators and volume profile
                df_indicators = self._calculate_volume_indicators(df.copy())
                
                # Build volume profile
                self._build_volume_profile(symbol, df_indicators)
                
                # Generate signals based on volume profile analysis
                symbol_signals = self._generate_volume_signals(symbol, df_indicators)
                signals.extend(symbol_signals)
                
            except Exception as e:
                print(f"Error generating volume profile signals for {symbol}: {e}")
                continue
                
        return signals
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-related indicators
        """
        # Volume moving averages
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # Volume ratio
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Average True Range
        df['ATR'] = self._calculate_atr(df, 14)
        
        # Price-Volume Trend
        df['PVT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()
        
        # Volume-weighted returns
        df['VW_Return'] = (df['Close'].pct_change() * df['Volume_Ratio']).rolling(window=5).mean()
        
        # Money Flow Index
        df['MFI'] = self._calculate_mfi(df, 14)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = pd.Series(index=df.index, dtype=float).fillna(0)
        negative_flow = pd.Series(index=df.index, dtype=float).fillna(0)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi
    
    def _build_volume_profile(self, symbol: str, df: pd.DataFrame):
        """
        Build volume profile for the given period
        """
        recent_df = df.tail(self.parameters['profile_period'])
        
        if len(recent_df) < 10:
            return
        
        # Calculate price range
        price_min = recent_df['Low'].min()
        price_max = recent_df['High'].max()
        
        # Create price bins
        bins = np.linspace(price_min, price_max, self.parameters['price_bins'])
        bin_width = bins[1] - bins[0]
        
        # Initialize volume profile
        volume_profile = np.zeros(len(bins) - 1)
        
        # Distribute volume across price levels
        for idx, row in recent_df.iterrows():
            # Simple distribution: assign volume proportionally to OHLC within the range
            prices = [row['Open'], row['High'], row['Low'], row['Close']]
            volume_per_price = row['Volume'] / 4
            
            for price in prices:
                bin_idx = np.digitize(price, bins) - 1
                if 0 <= bin_idx < len(volume_profile):
                    volume_profile[bin_idx] += volume_per_price
        
        # Store results
        self.volume_profiles[symbol] = {
            'bins': bins,
            'volumes': volume_profile,
            'timestamp': df.index[-1]
        }
        
        # Calculate Point of Control (POC) - highest volume bin
        poc_idx = np.argmax(volume_profile)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        self.poc_levels[symbol] = poc_price
        
        # Calculate Value Area
        self._calculate_value_area(symbol, bins, volume_profile)
    
    def _calculate_value_area(self, symbol: str, bins: np.ndarray, volume_profile: np.ndarray):
        """
        Calculate Value Area (area containing X% of volume)
        """
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * (self.parameters['value_area_percent'] / 100)
        
        # Start from POC and expand outward
        poc_idx = np.argmax(volume_profile)
        accumulated_volume = volume_profile[poc_idx]
        
        lower_idx = poc_idx
        upper_idx = poc_idx
        
        while accumulated_volume < target_volume and (lower_idx > 0 or upper_idx < len(volume_profile) - 1):
            # Expand to the side with higher volume
            lower_volume = volume_profile[lower_idx - 1] if lower_idx > 0 else 0
            upper_volume = volume_profile[upper_idx + 1] if upper_idx < len(volume_profile) - 1 else 0
            
            if lower_volume >= upper_volume and lower_idx > 0:
                lower_idx -= 1
                accumulated_volume += volume_profile[lower_idx]
            elif upper_idx < len(volume_profile) - 1:
                upper_idx += 1
                accumulated_volume += volume_profile[upper_idx]
            else:
                break
        
        # Store Value Area
        self.value_areas[symbol] = {
            'high': (bins[upper_idx] + bins[upper_idx + 1]) / 2,
            'low': (bins[lower_idx] + bins[lower_idx + 1]) / 2,
            'volume_percent': accumulated_volume / total_volume * 100
        }
    
    def _generate_volume_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on volume profile analysis
        """
        signals = []
        
        if symbol not in self.volume_profiles or symbol not in self.value_areas:
            return signals
        
        for i in range(len(df) - 10, len(df)):
            if i < 20:
                continue
                
            current = df.iloc[i]
            recent_df = df.iloc[max(0, i-5):i+1]
            
            signal = self._analyze_volume_patterns(symbol, current, recent_df, i, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_volume_patterns(self, symbol: str, current: pd.Series, 
                                recent_df: pd.DataFrame, index: int, 
                                full_df: pd.DataFrame) -> Signal:
        """
        Analyze volume profile patterns for signal generation
        """
        current_price = current['Close']
        volume = current['Volume']
        
        poc_level = self.poc_levels.get(symbol)
        value_area = self.value_areas.get(symbol)
        
        if not poc_level or not value_area:
            return None
        
        signal_type = None
        signal_strength = 0
        reasons = []
        
        # Check POC bounce/rejection
        poc_distance = abs(current_price - poc_level) / current_price
        if poc_distance < self.parameters['poc_bounce_threshold']:
            # Near POC level
            if current['Volume_Ratio'] > 1.5:  # High volume
                if current_price > poc_level and current['Close'] > current['Open']:
                    signal_type = SignalType.BUY
                    signal_strength += 0.3
                    reasons.append("poc_bounce_bullish")
                elif current_price < poc_level and current['Close'] < current['Open']:
                    signal_type = SignalType.SELL
                    signal_strength += 0.3
                    reasons.append("poc_bounce_bearish")
        
        # Check Value Area breakout
        if current_price > value_area['high']:
            va_breakout_strength = (current_price - value_area['high']) / value_area['high']
            if (va_breakout_strength > self.parameters['va_breakout_threshold'] and 
                current['Volume_Ratio'] > 1.8):
                signal_type = SignalType.BUY
                signal_strength += 0.4
                reasons.append("value_area_breakout_bull")
        elif current_price < value_area['low']:
            va_breakdown_strength = (value_area['low'] - current_price) / value_area['low']
            if (va_breakdown_strength > self.parameters['va_breakout_threshold'] and 
                current['Volume_Ratio'] > 1.8):
                signal_type = SignalType.SELL
                signal_strength += 0.4
                reasons.append("value_area_breakdown_bear")
        
        # Check volume imbalance
        avg_volume = recent_df['Volume'].mean()
        if volume > avg_volume * self.parameters['volume_imbalance_ratio']:
            if current['Close'] > current['Open']:
                if signal_type is None or signal_type == SignalType.BUY:
                    signal_type = SignalType.BUY
                    signal_strength += 0.2
                    reasons.append("volume_imbalance_bullish")
            else:
                if signal_type is None or signal_type == SignalType.SELL:
                    signal_type = SignalType.SELL
                    signal_strength += 0.2
                    reasons.append("volume_imbalance_bearish")
        
        # VWAP confirmation
        if current_price > current['VWAP'] and signal_type == SignalType.BUY:
            signal_strength += 0.1
            reasons.append("above_vwap")
        elif current_price < current['VWAP'] and signal_type == SignalType.SELL:
            signal_strength += 0.1
            reasons.append("below_vwap")
        
        # MFI confirmation
        mfi = current['MFI']
        if signal_type == SignalType.BUY and mfi > 50:
            signal_strength += 0.1
            reasons.append("mfi_bullish")
        elif signal_type == SignalType.SELL and mfi < 50:
            signal_strength += 0.1
            reasons.append("mfi_bearish")
        
        # Only generate signal if we have sufficient strength
        if signal_strength < 0.5 or signal_type is None:
            return None
        
        # Cap signal strength
        signal_strength = min(signal_strength, 1.0)
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=signal_strength,
            price=current_price,
            timestamp=current.name,
            metadata={
                'strategy': 'volume_profile',
                'reasons': reasons,
                'poc_level': poc_level,
                'value_area_high': value_area['high'],
                'value_area_low': value_area['low'],
                'volume_ratio': current['Volume_Ratio'],
                'mfi': mfi,
                'vwap': current['VWAP'],
                'atr': current['ATR']
            }
        )
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for volume profile trades
        """
        # Base position size
        base_size = 0.08  # 8% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on volume confirmation
        volume_ratio = signal.metadata.get('volume_ratio', 1)
        volume_multiplier = 1 + min((volume_ratio - 1) * 0.1, 0.3)  # Up to 30% increase
        
        # Adjust based on distance from key levels
        poc_level = signal.metadata.get('poc_level')
        if poc_level:
            price_distance = abs(signal.price - poc_level) / signal.price
            if price_distance < 0.01:  # Very close to POC
                distance_multiplier = 1.2  # 20% increase for POC trades
            else:
                distance_multiplier = 1.0
        else:
            distance_multiplier = 1.0
        
        # Volatility adjustment
        vol_multiplier = max(0.5, min(1.5, 1 / (1 + volatility * 2)))
        
        # Calculate final position size
        position_size = (base_size * strength_multiplier * volume_multiplier * 
                        distance_multiplier * vol_multiplier)
        
        # Cap at maximum position size
        max_position = 0.12  # Maximum 12% per position
        position_size = min(position_size, max_position)
        
        return position_size * portfolio_value / signal.price

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate a single signal for demo trading (adapter for generate_signals)
        """
        try:
            # Create a temporary data dict for compatibility
            symbol = 'TEMP'
            temp_data = {symbol: data}
            
            # Debug: Check data length and recent values
            print(f"        🔍 Volume Profile: Data length: {len(data)}")
            if len(data) >= 50:
                # Calculate volume profile for debugging
                current_price = data['Close'].iloc[-1]
                print(f"        🔍 Current price: {current_price:.2f}")
                
                # Calculate basic volume profile
                lookback = min(100, len(data))
                recent_data = data.iloc[-lookback:].copy()
                volume_profile = self._calculate_volume_profile(recent_data)
                
                if volume_profile:
                    poc_price = volume_profile['poc_price']
                    value_area_high = volume_profile['value_area_high']
                    value_area_low = volume_profile['value_area_low']
                    
                    print(f"        🔍 POC price: {poc_price:.2f}")
                    print(f"        🔍 Value area: {value_area_low:.2f} - {value_area_high:.2f}")
                    print(f"        🔍 Price vs POC: {(current_price - poc_price)/poc_price*100:.2f}%")
            
            # Get signals using the main method
            signals = self.generate_signals(temp_data)
            
            if signals:
                signal = signals[0]
                metadata = signal.metadata or {}
                reasons = metadata.get('reasons', [])
                poc_level = metadata.get('poc_level', 'N/A')
                
                reason = f"volume_profile: {', '.join(reasons)}, POC: {poc_level:.2f}"
                
                print(f"        ✅ Generated signal: {signal.signal_type}, strength: {signal.strength:.2f}")
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            else:
                print(f"        ❌ No signals generated")
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No volume profile signal'}
                
        except Exception as e:
            print(f"        ❌ Error in Volume Profile generate_signal: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """
        Calculate volume profile for the given data
        Returns POC and value area information
        """
        try:
            if len(df) < 10:
                return None
            
            # Calculate price range
            price_min = df['Low'].min()
            price_max = df['High'].max()
            
            if price_min >= price_max:
                return None
            
            # Create price bins
            bins = np.linspace(price_min, price_max, self.parameters['price_bins'])
            bin_width = bins[1] - bins[0]
            
            # Initialize volume profile
            volume_profile = np.zeros(len(bins) - 1)
            
            # Distribute volume across price levels
            for idx, row in df.iterrows():
                # Simple distribution: assign volume proportionally to OHLC within the range
                prices = [row['Open'], row['High'], row['Low'], row['Close']]
                volume_per_price = row['Volume'] / 4
                
                for price in prices:
                    bin_idx = np.digitize(price, bins) - 1
                    if 0 <= bin_idx < len(volume_profile):
                        volume_profile[bin_idx] += volume_per_price
            
            # Calculate Point of Control (POC) - highest volume bin
            poc_idx = np.argmax(volume_profile)
            poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
            
            # Calculate Value Area (70% of volume around POC)
            total_volume = np.sum(volume_profile)
            target_volume = total_volume * (self.parameters['value_area_percent'] / 100)
            
            # Start from POC and expand outward
            accumulated_volume = volume_profile[poc_idx]
            lower_idx = poc_idx
            upper_idx = poc_idx
            
            while accumulated_volume < target_volume and (lower_idx > 0 or upper_idx < len(volume_profile) - 1):
                # Expand to the side with higher volume
                lower_volume = volume_profile[lower_idx - 1] if lower_idx > 0 else 0
                upper_volume = volume_profile[upper_idx + 1] if upper_idx < len(volume_profile) - 1 else 0
                
                if lower_volume >= upper_volume and lower_idx > 0:
                    lower_idx -= 1
                    accumulated_volume += volume_profile[lower_idx]
                elif upper_idx < len(volume_profile) - 1:
                    upper_idx += 1
                    accumulated_volume += volume_profile[upper_idx]
                else:
                    break
            
            return {
                'poc_price': poc_price,
                'value_area_high': (bins[upper_idx] + bins[upper_idx + 1]) / 2,
                'value_area_low': (bins[lower_idx] + bins[lower_idx + 1]) / 2,
                'total_volume': total_volume,
                'value_area_volume_percent': accumulated_volume / total_volume * 100,
                'bins': bins,
                'volumes': volume_profile
            }
            
        except Exception as e:
            print(f"Error calculating volume profile: {e}")
            return None
