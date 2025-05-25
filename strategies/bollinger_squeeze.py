"""
Bollinger Band Squeeze Strategy
Identifies low volatility periods followed by explosive moves
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class BollingerSqueezeStrategy(BaseStrategy):
    """
    Bollinger Band Squeeze strategy that identifies periods of low volatility
    followed by breakout moves
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'bb_period': 20,              # Bollinger Band period
            'bb_std': 2.0,                # Bollinger Band standard deviations
            'kc_period': 20,              # Keltner Channel period
            'kc_atr_mult': 1.5,           # Keltner Channel ATR multiplier
            'squeeze_threshold': 0.95,    # Threshold for squeeze detection
            'momentum_period': 12,        # Momentum calculation period
            'squeeze_min_periods': 6,     # Minimum periods in squeeze
            'breakout_volume_mult': 1.5,  # Volume multiplier for breakout
            'momentum_threshold': 0.5,    # Minimum momentum for signal
            'volatility_percentile': 20,  # Low volatility percentile
            'squeeze_exit_threshold': 1.05, # Threshold to exit squeeze
            'trend_filter_period': 50,    # Trend filter MA period
            'min_squeeze_count': 3,       # Minimum consecutive squeeze bars
            'stop_loss_atr_mult': 2.0,
            'take_profit_atr_mult': 4.0
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Bollinger Squeeze", default_params)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate Bollinger Band Squeeze signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['bb_period'] + 50:
                continue
                
            try:
                # Calculate squeeze indicators
                df_indicators = self._calculate_squeeze_indicators(df.copy())
                
                # Generate signals based on squeeze analysis
                symbol_signals = self._generate_squeeze_signals(symbol, df_indicators)
                signals.extend(symbol_signals)
                
            except Exception as e:
                print(f"Error generating Bollinger Squeeze signals for {symbol}: {e}")
                continue
                
        return signals
    
    def _calculate_squeeze_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands, Keltner Channels, and squeeze indicators
        """
        # Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']
        
        df['BB_MA'] = df['Close'].rolling(window=bb_period).mean()
        bb_rolling_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_MA'] + (bb_rolling_std * bb_std)
        df['BB_Lower'] = df['BB_MA'] - (bb_rolling_std * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_MA']
        
        # Keltner Channels
        kc_period = self.parameters['kc_period']
        kc_mult = self.parameters['kc_atr_mult']
        
        df['KC_MA'] = df['Close'].rolling(window=kc_period).mean()
        df['ATR'] = self._calculate_atr(df, kc_period)
        df['KC_Upper'] = df['KC_MA'] + (df['ATR'] * kc_mult)
        df['KC_Lower'] = df['KC_MA'] - (df['ATR'] * kc_mult)
        df['KC_Width'] = (df['KC_Upper'] - df['KC_Lower']) / df['KC_MA']
        
        # Squeeze detection (BB inside KC)
        df['Squeeze'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])
        df['Squeeze_Ratio'] = df['BB_Width'] / df['KC_Width']
        
        # Momentum oscillator
        df['Momentum'] = self._calculate_momentum(df)
        
        # Squeeze strength and duration
        df['Squeeze_Count'] = self._calculate_squeeze_count(df)
        df['Squeeze_Intensity'] = self._calculate_squeeze_intensity(df)
        
        # Volatility percentile
        df['Volatility_Percentile'] = self._calculate_volatility_percentile(df)
        
        # Trend filter
        trend_period = self.parameters['trend_filter_period']
        df['Trend_MA'] = df['Close'].rolling(window=trend_period).mean()
        df['Trend_Direction'] = np.where(df['Close'] > df['Trend_MA'], 1, -1)
        
        # Price position within bands
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['KC_Position'] = (df['Close'] - df['KC_Lower']) / (df['KC_Upper'] - df['KC_Lower'])
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Breakout detection
        df['BB_Breakout_Upper'] = df['Close'] > df['BB_Upper']
        df['BB_Breakout_Lower'] = df['Close'] < df['BB_Lower']
        df['KC_Breakout_Upper'] = df['Close'] > df['KC_Upper']
        df['KC_Breakout_Lower'] = df['Close'] < df['KC_Lower']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(true_range).rolling(window=period).mean()
    
    def _calculate_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum oscillator for squeeze strategy
        """
        period = self.parameters['momentum_period']
        
        # Initialize momentum series with NaN values
        momentum = pd.Series(np.nan, index=df.index)
        
        # Calculate momentum starting from the period index
        for i in range(period, len(df)):
            y = df['Close'].iloc[i-period:i].values
            x = np.arange(len(y))
            
            # Linear regression slope
            if len(x) > 1 and len(y) > 1:
                try:
                    slope = np.polyfit(x, y, 1)[0]
                    momentum.iloc[i] = slope
                except:
                    momentum.iloc[i] = 0
            else:
                momentum.iloc[i] = 0
        
        return momentum
    
    def _calculate_squeeze_count(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate consecutive squeeze periods
        """
        squeeze_count = []
        current_count = 0
        
        for squeeze in df['Squeeze']:
            if squeeze:
                current_count += 1
            else:
                current_count = 0
            squeeze_count.append(current_count)
        
        return pd.Series(squeeze_count, index=df.index)
    
    def _calculate_squeeze_intensity(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate squeeze intensity (how tight the bands are)
        """
        # Lower ratio = tighter squeeze
        return 1 / (df['Squeeze_Ratio'] + 0.001)  # Add small value to avoid division by zero
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate rolling volatility percentile
        """
        window = min(100, len(df) - 1)  # Adjust window size based on available data
        
        if window < 20:  # Need minimum data for volatility calculation
            return pd.Series(np.nan, index=df.index)
        
        volatility = df['Close'].pct_change().rolling(window=20).std()
        
        # Initialize percentiles series with NaN values
        percentiles = pd.Series(np.nan, index=df.index)
        
        # Calculate percentiles starting from the window index
        for i in range(window, len(volatility)):
            current_vol = volatility.iloc[i]
            if pd.isna(current_vol):
                continue
                
            window_vols = volatility.iloc[i-window:i].dropna()
            if len(window_vols) > 0:
                percentile = (window_vols < current_vol).sum() / len(window_vols) * 100
                percentiles.iloc[i] = percentile
        
        return percentiles
    
    def _generate_squeeze_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on squeeze analysis
        """
        signals = []
        
        for i in range(len(df) - 10, len(df)):
            if i < 50:  # Need enough data
                continue
                
            current = df.iloc[i]
            recent_df = df.iloc[max(0, i-20):i+1]
            
            signal = self._analyze_squeeze_breakout(symbol, current, recent_df, i, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_squeeze_breakout(self, symbol: str, current: pd.Series, 
                                recent_df: pd.DataFrame, index: int, 
                                full_df: pd.DataFrame) -> Signal:
        """
        Analyze squeeze conditions for breakout signals
        """
        # Check if we're coming out of a squeeze
        if current['Squeeze']:
            return None  # Still in squeeze, no signal
        
        # Check if we just exited a squeeze
        squeeze_count = current['Squeeze_Count']
        if squeeze_count > 0:
            return None  # Still counting squeeze periods
        
        # Look for recent squeeze exit
        lookback = min(5, len(recent_df))
        recent_squeeze = recent_df['Squeeze'].iloc[-lookback:].any()
        
        if not recent_squeeze:
            return None  # No recent squeeze
        
        # Check squeeze quality
        max_squeeze_count = recent_df['Squeeze_Count'].max()
        if max_squeeze_count < self.parameters['min_squeeze_count']:
            return None  # Squeeze wasn't long enough
        
        # Analyze breakout direction and strength
        signal_strength = 0
        signal_type = None
        reasons = []
        
        # 1. Momentum direction
        momentum = current['Momentum']
        if abs(momentum) > self.parameters['momentum_threshold']:
            if momentum > 0:
                signal_type = SignalType.BUY
                signal_strength += 0.3
                reasons.append("bullish_momentum")
            else:
                signal_type = SignalType.SELL
                signal_strength += 0.3
                reasons.append("bearish_momentum")
        else:
            return None  # Momentum not strong enough
        
        # 2. Bollinger Band breakout
        if current['BB_Breakout_Upper'] and signal_type == SignalType.BUY:
            signal_strength += 0.25
            reasons.append("bb_breakout_upper")
        elif current['BB_Breakout_Lower'] and signal_type == SignalType.SELL:
            signal_strength += 0.25
            reasons.append("bb_breakout_lower")
        
        # 3. Keltner Channel breakout (stronger signal)
        if current['KC_Breakout_Upper'] and signal_type == SignalType.BUY:
            signal_strength += 0.3
            reasons.append("kc_breakout_upper")
        elif current['KC_Breakout_Lower'] and signal_type == SignalType.SELL:
            signal_strength += 0.3
            reasons.append("kc_breakout_lower")
        
        # 4. Volume confirmation
        volume_ratio = current['Volume_Ratio']
        if volume_ratio > self.parameters['breakout_volume_mult']:
            signal_strength += 0.2
            reasons.append("high_breakout_volume")
        
        # 5. Volatility expansion
        vol_percentile = current['Volatility_Percentile']
        if not pd.isna(vol_percentile) and vol_percentile > 80:
            signal_strength += 0.15
            reasons.append("volatility_expansion")
        
        # 6. Trend alignment
        trend_direction = current['Trend_Direction']
        if ((signal_type == SignalType.BUY and trend_direction > 0) or
            (signal_type == SignalType.SELL and trend_direction < 0)):
            signal_strength += 0.1
            reasons.append("trend_aligned")
        
        # 7. Squeeze intensity bonus
        avg_squeeze_intensity = recent_df['Squeeze_Intensity'].mean()
        if avg_squeeze_intensity > 2:  # High squeeze intensity
            signal_strength += 0.1
            reasons.append("high_squeeze_intensity")
        
        # Only generate signal if strength is sufficient
        if signal_strength < 0.6:
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
                'strategy': 'bollinger_squeeze',
                'reasons': reasons,
                'momentum': momentum,
                'squeeze_count': max_squeeze_count,
                'squeeze_intensity': avg_squeeze_intensity,
                'volume_ratio': volume_ratio,
                'volatility_percentile': vol_percentile,
                'bb_width': current['BB_Width'],
                'kc_width': current['KC_Width'],
                'squeeze_ratio': current['Squeeze_Ratio'],
                'trend_direction': trend_direction
            }
        )
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for squeeze breakout trades
        """
        # Base position size (squeeze breakouts can be aggressive)
        base_size = 0.12  # 12% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on squeeze quality
        squeeze_count = signal.metadata.get('squeeze_count', 0)
        squeeze_multiplier = 1 + min(squeeze_count / 20, 0.3)  # Up to 30% increase
        
        # Adjust based on volume confirmation
        volume_ratio = signal.metadata.get('volume_ratio', 1)
        volume_multiplier = min(volume_ratio / self.parameters['breakout_volume_mult'], 1.5)
        
        # Adjust based on squeeze intensity
        squeeze_intensity = signal.metadata.get('squeeze_intensity', 1)
        intensity_multiplier = 1 + min(squeeze_intensity / 5, 0.2)  # Up to 20% increase
        
        # Calculate final position size
        position_size = (base_size * strength_multiplier * squeeze_multiplier * 
                        volume_multiplier * intensity_multiplier)
        
        # Cap at maximum position size
        max_position = 0.25  # Maximum 25% for squeeze breakout trades
        position_size = min(position_size, max_position)
        
        return position_size * portfolio_value / signal.price
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate a single signal for demo trading (adapter for generate_signals)
        """
        try:
            # Convert single symbol data to multi-symbol format expected by generate_signals
            temp_data = {'TEMP_SYMBOL': data}
            
            # Debug: Check data length and recent values
            print(f"        🔍 Bollinger Squeeze: Data length: {len(data)}")
            if len(data) >= 20:
                # Calculate basic indicators for debugging
                df = data.copy()
                df_indicators = self._calculate_squeeze_indicators(df)
                
                # Check current conditions
                current = df_indicators.iloc[-1]
                print(f"        🔍 Current squeeze state: {current['Squeeze']}")
                print(f"        🔍 Momentum: {current['Momentum']:.4f}")
                print(f"        🔍 BB breakout upper/lower: {current['BB_Breakout_Upper']}/{current['BB_Breakout_Lower']}")
                print(f"        🔍 KC breakout upper/lower: {current['KC_Breakout_Upper']}/{current['KC_Breakout_Lower']}")
                print(f"        🔍 Volume ratio: {current['Volume_Ratio']:.2f}")
                
                # Check for recent squeeze
                recent_squeeze = df_indicators['Squeeze'].iloc[-5:].any()
                max_squeeze_count = df_indicators['Squeeze_Count'].max()
                print(f"        🔍 Recent squeeze (last 5): {recent_squeeze}, max count: {max_squeeze_count}")
            
            signals = self.generate_signals(temp_data)
            
            if signals:
                signal = signals[0]  # Take the first signal
                metadata = signal.metadata or {}
                reasons = metadata.get('reasons', [])
                strategy_name = metadata.get('strategy', 'bollinger_squeeze')
                reason = f"{strategy_name}: {', '.join(reasons[:3])}, strength: {signal.strength:.2f}"
                
                print(f"        ✅ Generated signal: {signal.signal_type}, strength: {signal.strength:.2f}")
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            
            print(f"        ❌ No signals generated")
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No squeeze signal'}
            
        except Exception as e:
            print(f"        ❌ Error in Bollinger Squeeze generate_signal: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
