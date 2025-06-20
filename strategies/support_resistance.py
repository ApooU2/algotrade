"""
Support and Resistance Strategy
Identifies key support/resistance levels and trades bounces and breakouts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class SupportResistanceStrategy(BaseStrategy):
    """
    Support and Resistance strategy that identifies key levels and trades bounces/breakouts
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 50,        # Period to look back for S/R levels
            'min_touches': 2,             # Minimum touches to confirm S/R level
            'level_tolerance': 0.005,     # 0.5% tolerance for level clustering
            'bounce_confirmation': 3,     # Bars to confirm bounce
            'breakout_confirmation': 2,   # Bars to confirm breakout
            'volume_breakout_mult': 1.5,  # Volume multiplier for breakouts
            'rsi_period': 14,             # RSI period for overbought/oversold
            'rsi_overbought': 70,         # RSI overbought level
            'rsi_oversold': 30,           # RSI oversold level
            'atr_period': 14,             # ATR period
            'level_strength_threshold': 3, # Minimum level strength
            'recent_level_weight': 1.5,   # Weight for recent levels
            'fibonacci_levels': True,     # Use Fibonacci retracement levels
            'pivot_points': True,         # Use pivot points
            'stop_loss_atr_mult': 2.0,
            'take_profit_ratio': 2.5,     # Risk:reward ratio
            'min_level_age': 5,           # Minimum age of level in periods
            'max_level_age': 200          # Maximum age of level in periods
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Support Resistance", default_params)
        
        # Store identified levels
        self.support_levels = {}
        self.resistance_levels = {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate Support/Resistance signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['lookback_period'] + 20:
                continue
                
            try:
                # Calculate indicators
                df_indicators = self._calculate_sr_indicators(df.copy())
                
                # Identify support and resistance levels
                self._identify_sr_levels(symbol, df_indicators)
                
                # Generate signals based on S/R analysis
                symbol_signals = self._generate_sr_signals(symbol, df_indicators)
                signals.extend(symbol_signals)
                
            except Exception as e:
                print(f"Error generating S/R signals for {symbol}: {e}")
                continue
                
        return signals
    
    def _calculate_sr_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate support/resistance indicators
        """
        # Basic indicators
        df['RSI'] = self._calculate_rsi(df, self.parameters['rsi_period'])
        df['ATR'] = self._calculate_atr(df, self.parameters['atr_period'])
        
        # Swing highs and lows
        df['Swing_High'] = self._identify_swing_highs(df)
        df['Swing_Low'] = self._identify_swing_lows(df)
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Momentum'] = df['Close'].pct_change(5)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Pivot points (if enabled)
        if self.parameters['pivot_points']:
            df = self._calculate_pivot_points(df)
        
        # Distance from recent highs/lows
        df['Dist_From_High'] = (df['High'].rolling(window=20).max() - df['Close']) / df['Close']
        df['Dist_From_Low'] = (df['Close'] - df['Low'].rolling(window=20).min()) / df['Close']
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _identify_swing_highs(self, df: pd.DataFrame) -> pd.Series:
        """Identify swing highs"""
        swing_highs = pd.Series(False, index=df.index)
        
        for i in range(2, len(df) - 2):
            if (df['High'].iloc[i] > df['High'].iloc[i-1] and 
                df['High'].iloc[i] > df['High'].iloc[i+1] and
                df['High'].iloc[i] > df['High'].iloc[i-2] and 
                df['High'].iloc[i] > df['High'].iloc[i+2]):
                swing_highs.iloc[i] = True
        
        return swing_highs
    
    def _identify_swing_lows(self, df: pd.DataFrame) -> pd.Series:
        """Identify swing lows"""
        swing_lows = pd.Series(False, index=df.index)
        
        for i in range(2, len(df) - 2):
            if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and 
                df['Low'].iloc[i] < df['Low'].iloc[i+1] and
                df['Low'].iloc[i] < df['Low'].iloc[i-2] and 
                df['Low'].iloc[i] < df['Low'].iloc[i+2]):
                swing_lows.iloc[i] = True
        
        return swing_lows
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily pivot points"""
        # For daily data, use previous day's high, low, close
        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
        df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
        df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
        
        return df
    
    def _identify_sr_levels(self, symbol: str, df: pd.DataFrame):
        """
        Identify and update support and resistance levels
        """
        lookback = self.parameters['lookback_period']
        tolerance = self.parameters['level_tolerance']
        min_touches = self.parameters['min_touches']
        
        # Get recent data
        recent_data = df.tail(lookback)
        
        # Find swing highs and lows
        swing_highs = recent_data[recent_data['Swing_High']]
        swing_lows = recent_data[recent_data['Swing_Low']]
        
        # Initialize levels if not exists
        if symbol not in self.resistance_levels:
            self.resistance_levels[symbol] = []
        if symbol not in self.support_levels:
            self.support_levels[symbol] = []
        
        # Process resistance levels (from swing highs)
        resistance_candidates = swing_highs['High'].tolist()
        self.resistance_levels[symbol] = self._cluster_levels(
            resistance_candidates, tolerance, min_touches, recent_data.index[-1]
        )
        
        # Process support levels (from swing lows)
        support_candidates = swing_lows['Low'].tolist()
        self.support_levels[symbol] = self._cluster_levels(
            support_candidates, tolerance, min_touches, recent_data.index[-1]
        )
        
        # Add Fibonacci levels if enabled
        if self.parameters['fibonacci_levels']:
            self._add_fibonacci_levels(symbol, recent_data)
        
        # Add pivot point levels if enabled
        if self.parameters['pivot_points']:
            self._add_pivot_levels(symbol, recent_data)
    
    def _cluster_levels(self, candidates: List[float], tolerance: float, 
                       min_touches: int, current_time) -> List[Dict]:
        """
        Cluster price levels that are close together
        """
        if not candidates:
            return []
        
        candidates.sort()
        clusters = []
        current_cluster = [candidates[0]]
        
        for price in candidates[1:]:
            if abs(price - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= min_touches:
                    clusters.append({
                        'level': np.mean(current_cluster),
                        'touches': len(current_cluster),
                        'strength': len(current_cluster),
                        'last_touch': current_time,
                        'age': 0
                    })
                current_cluster = [price]
        
        # Don't forget the last cluster
        if len(current_cluster) >= min_touches:
            clusters.append({
                'level': np.mean(current_cluster),
                'touches': len(current_cluster),
                'strength': len(current_cluster),
                'last_touch': current_time,
                'age': 0
            })
        
        return clusters
    
    def _add_fibonacci_levels(self, symbol: str, df: pd.DataFrame):
        """
        Add Fibonacci retracement levels
        """
        if len(df) < 20:
            return
        
        # Find recent significant high and low
        high = df['High'].tail(50).max()
        low = df['Low'].tail(50).min()
        
        # Calculate Fibonacci levels
        diff = high - low
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            level_price = high - ratio * diff
            
            # Only include levels between high and low
            if low <= level_price <= high:
                level_type = 'support' if level_price < df['Close'].iloc[-1] else 'resistance'
                self.support_levels[symbol].append({
                    'level': level_price,
                    'touches': 1,
                    'strength': 2.5,  # Fibonacci levels get moderate strength
                    'last_touch': df.index[-1],
                    'age': 0,
                    'type': f'fibonacci_{ratio:.1%}'
                })
    
    def _add_pivot_levels(self, symbol: str, df: pd.DataFrame):
        """
        Add pivot point levels
        """
        if 'Pivot' not in df.columns:
            return
        
        current_price = df['Close'].iloc[-1]
        latest = df.iloc[-1]
        
        # Add pivot levels as support/resistance
        pivot_levels = [
            ('Pivot', latest['Pivot']),
            ('R1', latest['R1']),
            ('S1', latest['S1']),
            ('R2', latest['R2']),
            ('S2', latest['S2'])
        ]
        
        for name, level in pivot_levels:
            if pd.isna(level):
                continue
                
            if level < current_price:
                self.support_levels[symbol].append({
                    'level': level,
                    'touches': 1,
                    'strength': 2,
                    'last_touch': df.index[-1],
                    'age': 0,
                    'type': f'pivot_{name}'
                })
            else:
                self.resistance_levels[symbol].append({
                    'level': level,
                    'touches': 1,
                    'strength': 2,
                    'last_touch': df.index[-1],
                    'age': 0,
                    'type': f'pivot_{name}'
                })
    
    def _generate_sr_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on S/R analysis
        """
        signals = []
        
        for i in range(len(df) - 10, len(df)):
            if i < 20:
                continue
                
            current = df.iloc[i]
            recent_df = df.iloc[max(0, i-10):i+1]
            
            signal = self._analyze_sr_patterns(symbol, current, recent_df, i, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_sr_patterns(self, symbol: str, current: pd.Series, 
                           recent_df: pd.DataFrame, index: int, 
                           full_df: pd.DataFrame) -> Signal:
        """
        Analyze support/resistance patterns for signal generation
        """
        if symbol not in self.support_levels or symbol not in self.resistance_levels:
            return None
        
        current_price = current['Close']
        signal_strength = 0
        signal_type = None
        reasons = []
        
        # Find nearest support and resistance levels
        nearest_support = self._find_nearest_level(current_price, self.support_levels[symbol], 'below')
        nearest_resistance = self._find_nearest_level(current_price, self.resistance_levels[symbol], 'above')
        
        # Check for bounces from support/resistance
        bounce_signal = self._check_bounce_pattern(current, recent_df, nearest_support, nearest_resistance)
        if bounce_signal:
            signal_type, strength, reason = bounce_signal
            signal_strength += strength
            reasons.append(reason)
        
        # Check for breakout patterns
        breakout_signal = self._check_breakout_pattern(current, recent_df, nearest_support, nearest_resistance)
        if breakout_signal:
            signal_type, strength, reason = breakout_signal
            signal_strength += strength
            reasons.append(reason)
        
        # RSI confirmation for bounces
        rsi = current['RSI']
        if signal_type == SignalType.BUY and rsi < self.parameters['rsi_oversold']:
            signal_strength += 0.2
            reasons.append("rsi_oversold")
        elif signal_type == SignalType.SELL and rsi > self.parameters['rsi_overbought']:
            signal_strength += 0.2
            reasons.append("rsi_overbought")
        
        # Volume confirmation
        volume_ratio = current['Volume_Ratio']
        if volume_ratio > 1.2:
            signal_strength += 0.15
            reasons.append("volume_confirmation")
        
        # Price momentum confirmation
        momentum = current['Price_Momentum']
        if signal_type == SignalType.BUY and momentum > 0:
            signal_strength += 0.1
            reasons.append("positive_momentum")
        elif signal_type == SignalType.SELL and momentum < 0:
            signal_strength += 0.1
            reasons.append("negative_momentum")
        
        # Only generate signal if we have sufficient strength
        if signal_strength < 0.5 or signal_type is None:
            return None
        
        # Cap signal strength
        signal_strength = min(signal_strength, 1.0)
        
        # Determine the key level being traded
        key_level = nearest_support if signal_type == SignalType.BUY else nearest_resistance
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=signal_strength,
            price=current_price,
            timestamp=current.name,
            metadata={
                'strategy': 'support_resistance',
                'reasons': reasons,
                'key_level': key_level['level'] if key_level else None,
                'level_strength': key_level['strength'] if key_level else None,
                'level_touches': key_level['touches'] if key_level else None,
                'nearest_support': nearest_support['level'] if nearest_support else None,
                'nearest_resistance': nearest_resistance['level'] if nearest_resistance else None,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'momentum': momentum,
                'atr': current['ATR']
            }
        )
    
    def _find_nearest_level(self, price: float, levels: List[Dict], direction: str) -> Dict:
        """
        Find the nearest support or resistance level
        """
        if not levels:
            return None
        
        valid_levels = []
        for level in levels:
            if direction == 'below' and level['level'] < price:
                valid_levels.append(level)
            elif direction == 'above' and level['level'] > price:
                valid_levels.append(level)
        
        if not valid_levels:
            return None
        
        # Sort by distance and return closest
        valid_levels.sort(key=lambda x: abs(x['level'] - price))
        return valid_levels[0]
    
    def _check_bounce_pattern(self, current: pd.Series, recent_df: pd.DataFrame, 
                            support: Dict, resistance: Dict) -> Tuple:
        """
        Check for bounce patterns from support/resistance
        """
        current_price = current['Close']
        tolerance = self.parameters['level_tolerance']
        
        # Check support bounce (buy signal)
        if support:
            distance_to_support = abs(current_price - support['level']) / support['level']
            if distance_to_support <= tolerance:
                # Check if price is moving up from support
                if len(recent_df) >= 3:
                    recent_lows = recent_df['Low'].tail(3)
                    if recent_lows.iloc[-1] > recent_lows.iloc[-3]:  # Higher lows
                        strength = 0.4 + (support['strength'] * 0.1)
                        return SignalType.BUY, strength, 'support_bounce'
        
        # Check resistance bounce (sell signal)
        if resistance:
            distance_to_resistance = abs(current_price - resistance['level']) / resistance['level']
            if distance_to_resistance <= tolerance:
                # Check if price is moving down from resistance
                if len(recent_df) >= 3:
                    recent_highs = recent_df['High'].tail(3)
                    if recent_highs.iloc[-1] < recent_highs.iloc[-3]:  # Lower highs
                        strength = 0.4 + (resistance['strength'] * 0.1)
                        return SignalType.SELL, strength, 'resistance_bounce'
        
        return None
    
    def _check_breakout_pattern(self, current: pd.Series, recent_df: pd.DataFrame, 
                              support: Dict, resistance: Dict) -> Tuple:
        """
        Check for breakout patterns through support/resistance
        """
        current_price = current['Close']
        volume_ratio = current['Volume_Ratio']
        
        # Check resistance breakout (buy signal)
        if resistance:
            if current_price > resistance['level']:
                # Confirm breakout with volume
                if volume_ratio > self.parameters['volume_breakout_mult']:
                    strength = 0.5 + (resistance['strength'] * 0.1)
                    return SignalType.BUY, strength, 'resistance_breakout'
        
        # Check support breakdown (sell signal)
        if support:
            if current_price < support['level']:
                # Confirm breakdown with volume
                if volume_ratio > self.parameters['volume_breakout_mult']:
                    strength = 0.5 + (support['strength'] * 0.1)
                    return SignalType.SELL, strength, 'support_breakdown'
        
        return None
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for S/R trades
        """
        # Base position size
        base_size = 0.10  # 10% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on level strength
        level_strength = signal.metadata.get('level_strength', 1)
        level_multiplier = 1 + min(level_strength / 5, 0.3)  # Up to 30% increase
        
        # Adjust based on signal type (bounces vs breakouts)
        reasons = signal.metadata.get('reasons', [])
        if any('breakout' in reason for reason in reasons):
            breakout_multiplier = 1.2  # Breakouts get 20% more
        else:
            breakout_multiplier = 1.0
        
        # Adjust based on volume confirmation
        volume_ratio = signal.metadata.get('volume_ratio', 1)
        volume_multiplier = min(volume_ratio / 1.2, 1.5)
        
        # Calculate final position size
        position_size = (base_size * strength_multiplier * level_multiplier * 
                        breakout_multiplier * volume_multiplier)
        
        # Cap at maximum position size
        max_position = 0.20  # Maximum 20% for S/R trades
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
            print(f"        🔍 Support/Resistance: Data length: {len(data)}")
            if len(data) >= 50:
                # Calculate basic indicators for debugging
                df = data.copy()
                levels = self._identify_support_resistance_levels(df)
                
                current_price = df['Close'].iloc[-1]
                print(f"        🔍 Current price: {current_price:.2f}")
                print(f"        🔍 Support levels found: {len([l for l in levels if l['type'] == 'support'])}")
                print(f"        🔍 Resistance levels found: {len([l for l in levels if l['type'] == 'resistance'])}")
                
                # Find nearest levels
                support_levels = [l for l in levels if l['type'] == 'support' and l['price'] < current_price]
                resistance_levels = [l for l in levels if l['type'] == 'resistance' and l['price'] > current_price]
                
                if support_levels:
                    nearest_support = max(support_levels, key=lambda x: x['price'])
                    print(f"        🔍 Nearest support: {nearest_support['price']:.2f} (strength: {nearest_support['strength']:.2f})")
                
                if resistance_levels:
                    nearest_resistance = min(resistance_levels, key=lambda x: x['price'])
                    print(f"        🔍 Nearest resistance: {nearest_resistance['price']:.2f} (strength: {nearest_resistance['strength']:.2f})")
            
            signals = self.generate_signals(temp_data)
            
            if signals:
                signal = signals[0] # Take the first signal
                metadata = signal.metadata or {}
                reasons = metadata.get('reasons', [])
                strategy_name = metadata.get('strategy', 'support_resistance')
                reason = f"{strategy_name}: {', '.join(reasons[:3])}, strength: {signal.strength:.2f}"
                
                print(f"        ✅ Generated signal: {signal.signal_type}, strength: {signal.strength:.2f}")
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            
            print(f"        ❌ No signals generated")
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No S/R signal'}
            
        except Exception as e:
            print(f"        ❌ Error in Support/Resistance generate_signal: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def _identify_support_resistance_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify support and resistance levels from price data
        Returns a list of levels with their properties
        """
        try:
            if len(df) < 50:
                return []
            
            lookback = min(self.parameters['lookback_period'], len(df) - 1)
            tolerance = self.parameters['level_tolerance']
            
            # Calculate indicators first
            df_with_indicators = self._calculate_sr_indicators(df.copy())
            
            # Find swing highs and lows
            recent_data = df_with_indicators.tail(lookback)
            swing_highs = recent_data[recent_data['Swing_High']]
            swing_lows = recent_data[recent_data['Swing_Low']]
            
            levels = []
            
            # Process resistance levels (from swing highs)
            if not swing_highs.empty:
                resistance_candidates = swing_highs['High'].tolist()
                resistance_clusters = self._cluster_levels(
                    resistance_candidates, tolerance, self.parameters['min_touches'], recent_data.index[-1]
                )
                
                for cluster in resistance_clusters:
                    levels.append({
                        'price': cluster['level'],
                        'type': 'resistance',
                        'strength': cluster['strength'],
                        'touches': cluster['touches'],
                        'age': cluster['age']
                    })
            
            # Process support levels (from swing lows)
            if not swing_lows.empty:
                support_candidates = swing_lows['Low'].tolist()
                support_clusters = self._cluster_levels(
                    support_candidates, tolerance, self.parameters['min_touches'], recent_data.index[-1]
                )
                
                for cluster in support_clusters:
                    levels.append({
                        'price': cluster['level'],
                        'type': 'support',
                        'strength': cluster['strength'],
                        'touches': cluster['touches'],
                        'age': cluster['age']
                    })
            
            # Add Fibonacci levels if enabled
            if self.parameters['fibonacci_levels']:
                fib_levels = self._calculate_fibonacci_levels(recent_data)
                levels.extend(fib_levels)
            
            # Add pivot point levels if enabled
            if self.parameters['pivot_points'] and 'Pivot' in df_with_indicators.columns:
                pivot_levels = self._calculate_pivot_point_levels(df_with_indicators)
                levels.extend(pivot_levels)
            
            # Filter levels by strength threshold
            min_strength = self.parameters.get('level_strength_threshold', 2)
            filtered_levels = [l for l in levels if l['strength'] >= min_strength]
            
            # Sort by strength (strongest first)
            filtered_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return filtered_levels
            
        except Exception as e:
            print(f"Error in _identify_support_resistance_levels: {e}")
            return []
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculate Fibonacci retracement levels
        """
        if len(df) < 20:
            return []
        
        # Find recent significant high and low
        high = df['High'].tail(50).max()
        low = df['Low'].tail(50).min()
        current_price = df['Close'].iloc[-1]
        
        # Calculate Fibonacci levels
        diff = high - low
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        levels = []
        for ratio in fib_ratios:
            level_price = high - ratio * diff
            
            # Only include levels between high and low
            if low <= level_price <= high:
                level_type = 'support' if level_price < current_price else 'resistance'
                levels.append({
                    'price': level_price,
                    'type': level_type,
                    'strength': 2.5,  # Fibonacci levels get moderate strength
                    'touches': 1,
                    'age': 0,
                    'source': f'fibonacci_{ratio:.1%}'
                })
        
        return levels
    
    def _calculate_pivot_point_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculate pivot point levels
        """
        if 'Pivot' not in df.columns or len(df) < 2:
            return []
        
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        levels = []
        pivot_data = [
            ('pivot', latest.get('Pivot')),
            ('r1', latest.get('R1')),
            ('s1', latest.get('S1')),
            ('r2', latest.get('R2')),
            ('s2', latest.get('S2'))
        ]
        
        for name, level_price in pivot_data:
            if pd.isna(level_price):
                continue
                
            level_type = 'support' if level_price < current_price else 'resistance'
            levels.append({
                'price': level_price,
                'type': level_type,
                'strength': 2.0,  # Pivot points get moderate strength
                'touches': 1,
                'age': 0,
                'source': f'pivot_{name}'
            })
        
        return levels

    # ...existing code...
