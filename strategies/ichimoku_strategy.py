"""
Ichimoku Cloud Strategy
Based on the Ichimoku Kinko Hyo system for trend identification and signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class IchimokuStrategy(BaseStrategy):
    """
    Ichimoku Cloud strategy using all components of the Ichimoku system
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'tenkan_period': 9,           # Tenkan-sen (Conversion Line)
            'kijun_period': 26,           # Kijun-sen (Base Line)
            'senkou_period': 52,          # Senkou Span B (Leading Span B)
            'displacement': 26,           # Cloud displacement
            'chikou_displacement': 26,    # Chikou Span displacement
            'cloud_breakout_confirm': True,  # Require cloud breakout confirmation
            'tk_cross_confirm': True,     # Require Tenkan-Kijun cross confirmation
            'price_above_cloud_confirm': True,  # Require price above cloud
            'chikou_confirm': True,       # Require Chikou confirmation
            'cloud_thickness_threshold': 0.02,  # 2% minimum cloud thickness
            'trend_filter_period': 100,   # Long-term trend filter
            'volume_confirmation': True,   # Require volume confirmation
            'min_volume_ratio': 1.3,     # Minimum volume ratio
            'stop_loss_atr_mult': 2.5,
            'take_profit_atr_mult': 4.0,
            'weak_signal_threshold': 0.4,  # Threshold for weak signals
            'strong_signal_threshold': 0.8  # Threshold for strong signals
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Ichimoku Cloud", default_params)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate Ichimoku Cloud signals
        """
        signals = []
        
        for symbol, df in data.items():
            required_length = max(
                self.parameters['senkou_period'], 
                self.parameters['displacement']
            ) + 50
            
            if len(df) < required_length:
                continue
                
            try:
                # Calculate Ichimoku indicators
                df_indicators = self._calculate_ichimoku_indicators(df.copy())
                
                # Generate signals based on Ichimoku analysis
                symbol_signals = self._generate_ichimoku_signals(symbol, df_indicators)
                signals.extend(symbol_signals)
                
            except Exception as e:
                print(f"Error generating Ichimoku signals for {symbol}: {e}")
                continue
                
        return signals
    
    def _calculate_ichimoku_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Ichimoku Cloud components
        """
        # Tenkan-sen (Conversion Line) - (9-period high + 9-period low) / 2
        tenkan_period = self.parameters['tenkan_period']
        df['Tenkan_Sen'] = (df['High'].rolling(window=tenkan_period).max() + 
                           df['Low'].rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line) - (26-period high + 26-period low) / 2
        kijun_period = self.parameters['kijun_period']
        df['Kijun_Sen'] = (df['High'].rolling(window=kijun_period).max() + 
                          df['Low'].rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A) - (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
        displacement = self.parameters['displacement']
        df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B) - (52-period high + 52-period low) / 2, plotted 26 periods ahead
        senkou_period = self.parameters['senkou_period']
        df['Senkou_Span_B'] = ((df['High'].rolling(window=senkou_period).max() + 
                               df['Low'].rolling(window=senkou_period).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span) - Close plotted 26 periods behind
        chikou_displacement = self.parameters['chikou_displacement']
        df['Chikou_Span'] = df['Close'].shift(-chikou_displacement)
        
        # Cloud analysis
        df['Cloud_Top'] = np.maximum(df['Senkou_Span_A'], df['Senkou_Span_B'])
        df['Cloud_Bottom'] = np.minimum(df['Senkou_Span_A'], df['Senkou_Span_B'])
        df['Cloud_Thickness'] = (df['Cloud_Top'] - df['Cloud_Bottom']) / df['Close']
        
        # Cloud color (bullish when Senkou A > Senkou B)
        df['Cloud_Bullish'] = df['Senkou_Span_A'] > df['Senkou_Span_B']
        
        # Price position relative to cloud
        df['Price_Above_Cloud'] = df['Close'] > df['Cloud_Top']
        df['Price_Below_Cloud'] = df['Close'] < df['Cloud_Bottom']
        df['Price_In_Cloud'] = ~df['Price_Above_Cloud'] & ~df['Price_Below_Cloud']
        
        # Tenkan-Kijun relationships
        df['TK_Cross_Up'] = (df['Tenkan_Sen'] > df['Kijun_Sen']) & (df['Tenkan_Sen'].shift(1) <= df['Kijun_Sen'].shift(1))
        df['TK_Cross_Down'] = (df['Tenkan_Sen'] < df['Kijun_Sen']) & (df['Tenkan_Sen'].shift(1) >= df['Kijun_Sen'].shift(1))
        df['Tenkan_Above_Kijun'] = df['Tenkan_Sen'] > df['Kijun_Sen']
        
        # Price vs Tenkan and Kijun
        df['Price_Above_Tenkan'] = df['Close'] > df['Tenkan_Sen']
        df['Price_Above_Kijun'] = df['Close'] > df['Kijun_Sen']
        
        # Cloud breakouts
        df['Cloud_Breakout_Up'] = (df['Close'] > df['Cloud_Top']) & (df['Close'].shift(1) <= df['Cloud_Top'].shift(1))
        df['Cloud_Breakout_Down'] = (df['Close'] < df['Cloud_Bottom']) & (df['Close'].shift(1) >= df['Cloud_Bottom'].shift(1))
        
        # Chikou analysis (look back to avoid future data)
        # Check if Chikou was above/below price 26 periods ago
        df['Chikou_Above_Price'] = df['Close'].shift(chikou_displacement) > df['Close']
        df['Chikou_Below_Price'] = df['Close'].shift(chikou_displacement) < df['Close']
        
        # Ichimoku strength analysis
        df['Ichimoku_Bullish_Count'] = (
            df['Price_Above_Cloud'].astype(int) +
            df['Tenkan_Above_Kijun'].astype(int) +
            df['Cloud_Bullish'].astype(int) +
            df['Price_Above_Tenkan'].astype(int) +
            df['Price_Above_Kijun'].astype(int) +
            df['Chikou_Above_Price'].astype(int)
        )
        
        # Trend strength
        df['Ichimoku_Trend_Strength'] = df['Ichimoku_Bullish_Count'] / 6
        
        # Volume analysis
        if self.parameters['volume_confirmation']:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Long-term trend filter
        trend_period = self.parameters['trend_filter_period']
        df['Long_Term_MA'] = df['Close'].rolling(window=trend_period).mean()
        df['Long_Term_Trend'] = np.where(df['Close'] > df['Long_Term_MA'], 1, -1)
        
        return df
    
    def _generate_ichimoku_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on Ichimoku analysis
        """
        signals = []
        
        for i in range(len(df) - 10, len(df)):
            if i < 100:  # Need enough data for all calculations
                continue
                
            current = df.iloc[i]
            recent_df = df.iloc[max(0, i-20):i+1]
            
            signal = self._analyze_ichimoku_patterns(symbol, current, recent_df, i, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_ichimoku_patterns(self, symbol: str, current: pd.Series, 
                                 recent_df: pd.DataFrame, index: int, 
                                 full_df: pd.DataFrame) -> Signal:
        """
        Analyze Ichimoku patterns for signal generation
        """
        signal_strength = 0
        signal_type = None
        reasons = []
        
        # Check if we have all required data
        required_fields = ['Tenkan_Sen', 'Kijun_Sen', 'Cloud_Top', 'Cloud_Bottom']
        if any(pd.isna(current[field]) for field in required_fields):
            return None
        
        # Volume confirmation
        if self.parameters['volume_confirmation']:
            if current['Volume_Ratio'] < self.parameters['min_volume_ratio']:
                return None
        
        # Cloud thickness filter
        if current['Cloud_Thickness'] < self.parameters['cloud_thickness_threshold']:
            return None  # Cloud too thin, signals may be unreliable
        
        # 1. Tenkan-Kijun Cross
        if self.parameters['tk_cross_confirm']:
            if current['TK_Cross_Up']:
                signal_type = SignalType.BUY
                signal_strength += 0.3
                reasons.append("tenkan_kijun_cross_up")
            elif current['TK_Cross_Down']:
                signal_type = SignalType.SELL
                signal_strength += 0.3
                reasons.append("tenkan_kijun_cross_down")
        
        # 2. Cloud breakout
        if self.parameters['cloud_breakout_confirm']:
            if current['Cloud_Breakout_Up'] and (signal_type == SignalType.BUY or signal_type is None):
                signal_type = SignalType.BUY
                signal_strength += 0.4
                reasons.append("cloud_breakout_up")
            elif current['Cloud_Breakout_Down'] and (signal_type == SignalType.SELL or signal_type is None):
                signal_type = SignalType.SELL
                signal_strength += 0.4
                reasons.append("cloud_breakout_down")
        
        # 3. Price position confirmations
        if self.parameters['price_above_cloud_confirm']:
            if current['Price_Above_Cloud'] and signal_type == SignalType.BUY:
                signal_strength += 0.2
                reasons.append("price_above_cloud")
            elif current['Price_Below_Cloud'] and signal_type == SignalType.SELL:
                signal_strength += 0.2
                reasons.append("price_below_cloud")
        
        # 4. Chikou confirmation
        if self.parameters['chikou_confirm']:
            if current['Chikou_Above_Price'] and signal_type == SignalType.BUY:
                signal_strength += 0.15
                reasons.append("chikou_bullish")
            elif current['Chikou_Below_Price'] and signal_type == SignalType.SELL:
                signal_strength += 0.15
                reasons.append("chikou_bearish")
        
        # 5. Ichimoku trend strength
        trend_strength = current['Ichimoku_Trend_Strength']
        if signal_type == SignalType.BUY and trend_strength > 0.6:
            signal_strength += 0.2
            reasons.append("strong_bullish_alignment")
        elif signal_type == SignalType.SELL and trend_strength < 0.4:
            signal_strength += 0.2
            reasons.append("strong_bearish_alignment")
        
        # 6. Cloud color confirmation
        if current['Cloud_Bullish'] and signal_type == SignalType.BUY:
            signal_strength += 0.1
            reasons.append("bullish_cloud")
        elif not current['Cloud_Bullish'] and signal_type == SignalType.SELL:
            signal_strength += 0.1
            reasons.append("bearish_cloud")
        
        # 7. Price vs Tenkan and Kijun
        if signal_type == SignalType.BUY:
            if current['Price_Above_Tenkan'] and current['Price_Above_Kijun']:
                signal_strength += 0.1
                reasons.append("price_above_tk")
        elif signal_type == SignalType.SELL:
            if not current['Price_Above_Tenkan'] and not current['Price_Above_Kijun']:
                signal_strength += 0.1
                reasons.append("price_below_tk")
        
        # 8. Long-term trend alignment
        long_term_trend = current['Long_Term_Trend']
        if ((signal_type == SignalType.BUY and long_term_trend > 0) or
            (signal_type == SignalType.SELL and long_term_trend < 0)):
            signal_strength += 0.1
            reasons.append("long_term_trend_aligned")
        
        # 9. Multiple Ichimoku confirmations
        bullish_count = current['Ichimoku_Bullish_Count']
        if signal_type == SignalType.BUY and bullish_count >= 5:
            signal_strength += 0.15
            reasons.append("multiple_bullish_confirmations")
        elif signal_type == SignalType.SELL and bullish_count <= 1:
            signal_strength += 0.15
            reasons.append("multiple_bearish_confirmations")
        
        # Only generate signal if we have a type and sufficient strength
        if signal_type is None or signal_strength < self.parameters['weak_signal_threshold']:
            return None
        
        # Cap signal strength
        signal_strength = min(signal_strength, 1.0)
        
        # Classify signal strength
        if signal_strength >= self.parameters['strong_signal_threshold']:
            signal_class = 'strong'
        elif signal_strength >= 0.6:
            signal_class = 'medium'
        else:
            signal_class = 'weak'
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=signal_strength,
            price=current['Close'],
            timestamp=current.name,
            metadata={
                'strategy': 'ichimoku',
                'signal_class': signal_class,
                'reasons': reasons,
                'tenkan_sen': current['Tenkan_Sen'],
                'kijun_sen': current['Kijun_Sen'],
                'cloud_top': current['Cloud_Top'],
                'cloud_bottom': current['Cloud_Bottom'],
                'cloud_thickness': current['Cloud_Thickness'],
                'trend_strength': trend_strength,
                'bullish_count': bullish_count,
                'cloud_bullish': current['Cloud_Bullish'],
                'long_term_trend': long_term_trend,
                'volume_ratio': current.get('Volume_Ratio', 1.0)
            }
        )
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for Ichimoku trades
        """
        # Base position size varies by signal strength
        signal_class = signal.metadata.get('signal_class', 'weak')
        
        if signal_class == 'strong':
            base_size = 0.15  # 15% for strong signals
        elif signal_class == 'medium':
            base_size = 0.12  # 12% for medium signals
        else:
            base_size = 0.08  # 8% for weak signals
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on trend strength
        trend_strength = signal.metadata.get('trend_strength', 0.5)
        if signal.signal_type == SignalType.BUY:
            trend_multiplier = 0.8 + (trend_strength * 0.4)  # 0.8 to 1.2
        else:
            trend_multiplier = 0.8 + ((1 - trend_strength) * 0.4)  # 0.8 to 1.2
        
        # Adjust based on cloud thickness (thicker cloud = stronger signal)
        cloud_thickness = signal.metadata.get('cloud_thickness', 0.02)
        thickness_multiplier = 1 + min(cloud_thickness * 10, 0.3)  # Up to 30% increase
        
        # Adjust based on confirmations
        bullish_count = signal.metadata.get('bullish_count', 3)
        confirmation_multiplier = 0.7 + (bullish_count / 6 * 0.6)  # 0.7 to 1.3
        
        # Calculate final position size
        position_size = (base_size * strength_multiplier * trend_multiplier * 
                        thickness_multiplier * confirmation_multiplier)
        
        # Cap at maximum position size
        max_position = 0.25  # Maximum 25% for Ichimoku trades
        position_size = min(position_size, max_position)
        
        return position_size * portfolio_value / signal.price

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate a single signal for demo trading (adapter for generate_signals)
        """
        try:
            # Debug: Check data length
            print(f"        🔍 Ichimoku: Data length: {len(data)}")
            
            # Need enough data for Ichimoku calculations
            required_length = max(
                self.parameters['senkou_period'], 
                self.parameters['displacement']
            ) + 50
            
            if len(data) < required_length:
                print(f"        🔍 Insufficient data for Ichimoku (need {required_length}+, have {len(data)})")
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Insufficient data for Ichimoku'}
            
            # Calculate basic indicators for debugging
            df = data.copy()
            df = self._calculate_ichimoku_indicators(df)
            
            # Check current conditions with correct column names
            current = df.iloc[-1]
            
            # Debug with correct column names
            if not pd.isna(current['Price_Above_Cloud']):
                print(f"        🔍 Price vs Cloud: above={current['Price_Above_Cloud']}, below={current['Price_Below_Cloud']}")
            if not pd.isna(current['Tenkan_Sen']):
                print(f"        🔍 Tenkan: {current['Tenkan_Sen']:.2f}, Kijun: {current['Kijun_Sen']:.2f}")
            if 'Chikou_Above_Price' in current and not pd.isna(current['Chikou_Above_Price']):
                print(f"        🔍 Chikou above price: {current['Chikou_Above_Price']}")
            if not pd.isna(current['Cloud_Thickness']):
                print(f"        🔍 Cloud thickness: {current['Cloud_Thickness']:.4f}")
            
            # Convert single symbol data to multi-symbol format expected by generate_signals
            temp_data = {'TEMP_SYMBOL': df}
            signals = self.generate_signals(temp_data)
            
            if signals:
                signal = signals[0]  # Take the first signal
                metadata = signal.metadata or {}
                reasons = metadata.get('reasons', [])
                strategy_name = metadata.get('strategy', 'ichimoku')
                reason = f"{strategy_name}: {', '.join(reasons[:3])}, strength: {signal.strength:.2f}"
                
                print(f"        ✅ Generated signal: {signal.signal_type}, strength: {signal.strength:.2f}")
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            
            print(f"        ❌ No signals generated")
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No Ichimoku signal'}
            
        except Exception as e:
            print(f"        ❌ Error in Ichimoku generate_signal: {e}")
            import traceback
            print(f"        🔍 Traceback: {traceback.format_exc()}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
