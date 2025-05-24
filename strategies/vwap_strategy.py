"""
VWAP (Volume Weighted Average Price) Strategy
Trades based on price relationship to VWAP and volume analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class VWAPStrategy(BaseStrategy):
    """
    VWAP strategy that analyzes price relationship to VWAP and volume patterns
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'vwap_periods': [9, 21, 50],  # Multiple VWAP timeframes
            'volume_threshold': 1.5,      # Volume must be 1.5x average
            'price_deviation_threshold': 0.005,  # 0.5% from VWAP
            'volume_profile_periods': 20,  # Periods for volume profile
            'vwap_momentum_threshold': 0.002,  # VWAP slope threshold
            'cumulative_vwap_periods': 50,  # Periods for cumulative VWAP
            'anchored_vwap_reset': 'daily',  # Reset frequency for anchored VWAP
            'stop_loss_atr_mult': 1.8,
            'take_profit_atr_mult': 3.5,
            'min_volume_ratio': 1.2,
            'vwap_band_width': 0.01  # 1% bands around VWAP
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("VWAP Strategy", default_params)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate VWAP-based trading signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < max(self.parameters['vwap_periods']) + 20:
                continue
                
            try:
                # Calculate VWAP indicators
                df_indicators = self._calculate_vwap_indicators(df.copy())
                
                # Generate signals based on VWAP analysis
                symbol_signals = self._generate_vwap_signals(symbol, df_indicators)
                signals.extend(symbol_signals)
                
            except Exception as e:
                print(f"Error generating VWAP signals for {symbol}: {e}")
                continue
                
        return signals
    
    def _calculate_vwap_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various VWAP indicators and volume metrics
        """
        # Standard VWAP
        df['VWAP'] = self._calculate_vwap(df, self.parameters['vwap_periods'][1])
        
        # Multiple timeframe VWAPs
        for period in self.parameters['vwap_periods']:
            df[f'VWAP_{period}'] = self._calculate_vwap(df, period)
        
        # VWAP bands (similar to Bollinger Bands)
        df['VWAP_Upper'] = df['VWAP'] * (1 + self.parameters['vwap_band_width'])
        df['VWAP_Lower'] = df['VWAP'] * (1 - self.parameters['vwap_band_width'])
        
        # VWAP momentum (slope)
        df['VWAP_Momentum'] = df['VWAP'].pct_change(5)
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Profile'] = self._calculate_volume_profile(df)
        
        # Price relative to VWAP
        df['Price_VWAP_Ratio'] = df['Close'] / df['VWAP']
        df['Price_VWAP_Deviation'] = (df['Close'] - df['VWAP']) / df['VWAP']
        
        # Anchored VWAP (reset daily)
        df['Anchored_VWAP'] = self._calculate_anchored_vwap(df)
        
        # VWAP convergence/divergence
        df['VWAP_Convergence'] = self._calculate_vwap_convergence(df)
        
        # Cumulative volume delta
        df['Cumulative_Volume_Delta'] = self._calculate_cumulative_volume_delta(df)
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Volume Weighted Average Price
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        volume_price = typical_price * df['Volume']
        
        return (volume_price.rolling(window=period).sum() / 
                df['Volume'].rolling(window=period).sum())
    
    def _calculate_anchored_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate anchored VWAP that resets at specified intervals
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Reset at the beginning of each day (if intraday data)
        # For daily data, this is just regular VWAP
        df_reset = df.copy()
        df_reset['Date'] = pd.to_datetime(df_reset.index).date
        
        anchored_vwap = []
        for date in df_reset['Date'].unique():
            date_mask = df_reset['Date'] == date
            date_data = df_reset[date_mask]
            
            if len(date_data) > 0:
                cumsum_volume_price = (typical_price[date_mask] * df['Volume'][date_mask]).cumsum()
                cumsum_volume = df['Volume'][date_mask].cumsum()
                day_vwap = cumsum_volume_price / cumsum_volume
                anchored_vwap.extend(day_vwap.tolist())
        
        return pd.Series(anchored_vwap, index=df.index)
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volume profile strength
        """
        period = self.parameters['volume_profile_periods']
        
        # Calculate volume at different price levels
        volume_profile = []
        
        for i in range(period, len(df)):
            window_data = df.iloc[i-period:i]
            
            # Divide price range into bins
            price_min = window_data['Low'].min()
            price_max = window_data['High'].max()
            
            if price_max == price_min:
                volume_profile.append(0.5)
                continue
            
            # Calculate volume-weighted price distribution
            typical_prices = (window_data['High'] + window_data['Low'] + window_data['Close']) / 3
            volumes = window_data['Volume']
            
            # Current price percentile in volume profile
            current_price = df.iloc[i]['Close']
            price_percentile = (current_price - price_min) / (price_max - price_min)
            
            volume_profile.append(price_percentile)
        
        # Pad with NaN for the initial period
        return pd.Series([np.nan] * period + volume_profile, index=df.index)
    
    def _calculate_vwap_convergence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate convergence/divergence between different VWAP timeframes
        """
        vwap_short = df[f'VWAP_{self.parameters["vwap_periods"][0]}']
        vwap_long = df[f'VWAP_{self.parameters["vwap_periods"][-1]}']
        
        return (vwap_short - vwap_long) / vwap_long
    
    def _calculate_cumulative_volume_delta(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate cumulative volume delta (approximation)
        """
        # Approximate buying vs selling pressure
        # Up volume when close > open, down volume when close < open
        up_volume = np.where(df['Close'] > df['Open'], df['Volume'], 0)
        down_volume = np.where(df['Close'] < df['Open'], df['Volume'], 0)
        
        volume_delta = up_volume - down_volume
        return pd.Series(volume_delta, index=df.index).rolling(window=10).sum()
    
    def _generate_vwap_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on VWAP analysis
        """
        signals = []
        
        for i in range(len(df) - 10, len(df)):
            if i < 50:  # Need enough data
                continue
                
            current = df.iloc[i]
            
            # Check basic requirements
            if (current['Volume_Ratio'] < self.parameters['min_volume_ratio'] or
                pd.isna(current['VWAP']) or
                pd.isna(current['VWAP_Momentum'])):
                continue
            
            signal = self._analyze_vwap_signals(symbol, current, df.iloc[max(0, i-20):i+1])
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_vwap_signals(self, symbol: str, current: pd.Series, recent_df: pd.DataFrame) -> Signal:
        """
        Analyze VWAP patterns for signal generation
        """
        # Debug: Check if required indicators exist
        required_cols = ['Price_VWAP_Deviation', 'VWAP_Momentum', 'Volume_Ratio', 'VWAP_Convergence']
        missing_cols = [col for col in required_cols if col not in current.index or pd.isna(current[col])]
        if missing_cols:
            print(f"      🔍 {symbol}: Missing VWAP indicators: {missing_cols}")
            return None

        # Signal strength accumulator
        signal_strength = 0
        signal_type = None
        reasons = []
        
        # 1. VWAP Mean Reversion
        price_deviation = current['Price_VWAP_Deviation']
        print(f"      🔍 {symbol}: VWAP deviation={price_deviation:.4f}, threshold={self.parameters['price_deviation_threshold']}")
        
        if abs(price_deviation) > self.parameters['price_deviation_threshold']:
            if price_deviation < -self.parameters['price_deviation_threshold']:
                # Price below VWAP - potential buy
                signal_strength += 0.3
                signal_type = SignalType.BUY
                reasons.append("price_below_vwap")
            elif price_deviation > self.parameters['price_deviation_threshold']:
                # Price above VWAP - potential sell
                signal_strength += 0.3
                signal_type = SignalType.SELL
                reasons.append("price_above_vwap")
        
        # 2. VWAP Momentum
        vwap_momentum = current['VWAP_Momentum']
        print(f"      🔍 {symbol}: VWAP momentum={vwap_momentum:.4f}, threshold={self.parameters['vwap_momentum_threshold']}")
        
        if abs(vwap_momentum) > self.parameters['vwap_momentum_threshold']:
            if vwap_momentum > 0:
                if signal_type == SignalType.BUY or signal_type is None:
                    signal_strength += 0.25
                    signal_type = SignalType.BUY
                    reasons.append("vwap_uptrend")
            else:
                if signal_type == SignalType.SELL or signal_type is None:
                    signal_strength += 0.25
                    signal_type = SignalType.SELL
                    reasons.append("vwap_downtrend")
        
        # 3. Volume Confirmation
        volume_ratio = current['Volume_Ratio']
        print(f"      🔍 {symbol}: Volume ratio={volume_ratio:.2f}, threshold={self.parameters['volume_threshold']}")
        
        if volume_ratio > self.parameters['volume_threshold']:
            signal_strength += 0.2
            reasons.append("high_volume")
        
        print(f"      🔍 {symbol}: Current signal strength={signal_strength:.2f}, type={signal_type}")
        
        # ...existing code for remaining analysis...
        
        # 4. VWAP Convergence
        convergence = current['VWAP_Convergence']
        if abs(convergence) > 0.001:  # 0.1% threshold
            if convergence > 0 and signal_type == SignalType.BUY:
                signal_strength += 0.15
                reasons.append("vwap_bullish_convergence")
            elif convergence < 0 and signal_type == SignalType.SELL:
                signal_strength += 0.15
                reasons.append("vwap_bearish_convergence")
        
        # 5. Volume Delta
        volume_delta = current['Cumulative_Volume_Delta']
        if not pd.isna(volume_delta):
            if volume_delta > 0 and signal_type == SignalType.BUY:
                signal_strength += 0.1
                reasons.append("positive_volume_delta")
            elif volume_delta < 0 and signal_type == SignalType.SELL:
                signal_strength += 0.1
                reasons.append("negative_volume_delta")
        
        # 6. VWAP Band Position
        if current['Close'] < current['VWAP_Lower'] and signal_type == SignalType.BUY:
            signal_strength += 0.15
            reasons.append("below_vwap_lower_band")
        elif current['Close'] > current['VWAP_Upper'] and signal_type == SignalType.SELL:
            signal_strength += 0.15
            reasons.append("above_vwap_upper_band")
        
        print(f"      🔍 {symbol}: Final signal strength={signal_strength:.2f}, minimum=0.5")
        
        # Only generate signal if strength is sufficient
        if signal_strength < 0.5 or signal_type is None:
            return None
        
        # Cap signal strength
        signal_strength = min(signal_strength, 1.0)
        
        print(f"      📊 {symbol}: VWAP signal generated - {signal_type.name} with strength {signal_strength:.2f}")
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=signal_strength,
            price=current['Close'],
            timestamp=current.name,
            metadata={
                'strategy': 'vwap',
                'reasons': reasons,
                'vwap_deviation': price_deviation,
                'vwap_momentum': vwap_momentum,
                'volume_ratio': volume_ratio,
                'vwap_convergence': convergence,
                'volume_delta': volume_delta,
                'current_vwap': current['VWAP'],
                'anchored_vwap': current['Anchored_VWAP']
            }
        )
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for VWAP trades
        """
        # Base position size
        base_size = 0.08  # 8% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on volume confirmation
        volume_ratio = signal.metadata.get('volume_ratio', 1)
        volume_multiplier = min(volume_ratio / self.parameters['volume_threshold'], 1.5)
        
        # Adjust based on VWAP deviation (higher deviation = higher conviction)
        vwap_deviation = abs(signal.metadata.get('vwap_deviation', 0))
        deviation_multiplier = 1 + min(vwap_deviation * 10, 0.3)  # Up to 30% increase
        
        # Calculate final position size
        position_size = base_size * strength_multiplier * volume_multiplier * deviation_multiplier
        
        # Cap at maximum position size
        max_position = 0.20  # Maximum 20% for VWAP trades
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
                metadata = signal.metadata or {}
                reasons = metadata.get('reasons', [])
                strategy_name = metadata.get('strategy', 'vwap')
                reason = f"{strategy_name}: {', '.join(reasons[:3])}, strength: {signal.strength:.2f}"
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No VWAP signal'}
            
        except Exception as e:
            print(f"Error in VWAP generate_signal: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
