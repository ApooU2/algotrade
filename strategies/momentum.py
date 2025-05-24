"""
Momentum Strategy
Captures trends and momentum in price movements
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal, SignalType

class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using multiple timeframe analysis and trend indicators
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'short_ma': 12,
            'long_ma': 26,
            'signal_ma': 9,
            'rsi_period': 14,
            'rsi_momentum_threshold': 55,
            'volume_ma': 20,
            'min_volume_ratio': 1.3,
            'min_momentum_threshold': 0.02,  # 2% minimum momentum
            'atr_period': 14,
            'stop_loss_atr_mult': 2,
            'take_profit_atr_mult': 4
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Momentum", default_params)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate momentum-based trading signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['long_ma']:
                continue
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Get latest data
            latest = df.iloc[-1]
            recent = df.tail(10)
            
            # Check momentum conditions
            signal = self._check_momentum_conditions(symbol, latest, recent, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators
        """
        # Moving averages
        df['EMA_short'] = df['Close'].ewm(span=self.parameters['short_ma']).mean()
        df['EMA_long'] = df['Close'].ewm(span=self.parameters['long_ma']).mean()
        
        # MACD
        df['MACD'] = df['EMA_short'] - df['EMA_long']
        df['MACD_signal'] = df['MACD'].ewm(span=self.parameters['signal_ma']).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], self.parameters['rsi_period'])
        
        # Price momentum (rate of change)
        df['Momentum_5'] = df['Close'].pct_change(5)
        df['Momentum_10'] = df['Close'].pct_change(10)
        df['Momentum_20'] = df['Close'].pct_change(20)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=self.parameters['volume_ma']).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # ATR for position sizing
        df['ATR'] = self._calculate_atr(df, self.parameters['atr_period'])
        
        # Trend strength
        df['Trend_strength'] = abs(df['EMA_short'] - df['EMA_long']) / df['Close']
        
        # Price acceleration
        df['Price_acceleration'] = df['Close'].pct_change().diff()
        
        # Higher high / Lower low patterns
        df['Higher_high'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['Lower_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
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
    
    def _check_momentum_conditions(self, symbol: str, latest: pd.Series, 
                                 recent: pd.DataFrame, df: pd.DataFrame) -> Signal:
        """
        Check for momentum entry conditions
        """
        # Bullish momentum conditions
        bullish_conditions = [
            latest['EMA_short'] > latest['EMA_long'],  # Short MA above long MA
            latest['MACD'] > latest['MACD_signal'],  # MACD above signal
            latest['MACD_histogram'] > recent['MACD_histogram'].iloc[-2],  # MACD histogram increasing
            latest['RSI'] > self.parameters['rsi_momentum_threshold'],  # RSI shows momentum
            latest['RSI'] < 80,  # But not overbought
            latest['Momentum_5'] > self.parameters['min_momentum_threshold'],  # Positive 5-day momentum
            latest['Momentum_10'] > 0,  # Positive 10-day momentum
            latest['Volume_ratio'] > self.parameters['min_volume_ratio'],  # Higher volume
            latest['Price_acceleration'] > 0,  # Accelerating upward
            recent['Higher_high'].any()  # Recent higher highs pattern
        ]
        
        # Bearish momentum conditions
        bearish_conditions = [
            latest['EMA_short'] < latest['EMA_long'],  # Short MA below long MA
            latest['MACD'] < latest['MACD_signal'],  # MACD below signal
            latest['MACD_histogram'] < recent['MACD_histogram'].iloc[-2],  # MACD histogram decreasing
            latest['RSI'] < (100 - self.parameters['rsi_momentum_threshold']),  # RSI shows bearish momentum
            latest['RSI'] > 20,  # But not oversold
            latest['Momentum_5'] < -self.parameters['min_momentum_threshold'],  # Negative 5-day momentum
            latest['Momentum_10'] < 0,  # Negative 10-day momentum
            latest['Volume_ratio'] > self.parameters['min_volume_ratio'],  # Higher volume
            latest['Price_acceleration'] < 0,  # Accelerating downward
            recent['Lower_low'].any()  # Recent lower lows pattern
        ]
        
        # Calculate signal strength
        bullish_strength = sum(bullish_conditions) / len(bullish_conditions)
        bearish_strength = sum(bearish_conditions) / len(bearish_conditions)
        
        # Additional strength factors
        trend_strength_factor = latest['Trend_strength'] * 10  # Scale trend strength
        volume_factor = min(latest['Volume_ratio'] / self.parameters['min_volume_ratio'], 2.0)
        
        if bullish_strength >= 0.7:  # At least 70% of conditions met
            strength = bullish_strength * (1 + trend_strength_factor * 0.3) * (1 + volume_factor * 0.2)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=min(strength, 1.0),
                price=latest['Close'],
                timestamp=latest.name,
                metadata={
                    'strategy': 'momentum',
                    'macd': latest['MACD'],
                    'rsi': latest['RSI'],
                    'momentum_5': latest['Momentum_5'],
                    'trend_strength': latest['Trend_strength'],
                    'volume_ratio': latest['Volume_ratio'],
                    'atr': latest['ATR']
                }
            )
        
        elif bearish_strength >= 0.7:  # At least 70% of conditions met
            strength = bearish_strength * (1 + trend_strength_factor * 0.3) * (1 + volume_factor * 0.2)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=min(strength, 1.0),
                price=latest['Close'],
                timestamp=latest.name,
                metadata={
                    'strategy': 'momentum',
                    'macd': latest['MACD'],
                    'rsi': latest['RSI'],
                    'momentum_5': latest['Momentum_5'],
                    'trend_strength': latest['Trend_strength'],
                    'volume_ratio': latest['Volume_ratio'],
                    'atr': latest['ATR']
                }
            )
        
        return None
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for momentum trades
        """
        # Base position size
        base_size = 0.08  # 8% base allocation for momentum trades
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on trend strength
        trend_strength = signal.metadata.get('trend_strength', 0)
        trend_multiplier = 1 + min(trend_strength * 5, 0.5)  # Up to 50% increase for strong trends
        
        # Adjust based on momentum magnitude
        momentum = abs(signal.metadata.get('momentum_5', 0))
        momentum_multiplier = 1 + min(momentum * 10, 0.3)  # Up to 30% increase for strong momentum
        
        # Volatility adjustment (momentum strategies can handle higher vol)
        vol_adjustment = max(0.7, min(1.5, 0.15 / volatility))
        
        # Calculate final position size
        position_size = base_size * strength_multiplier * trend_multiplier * momentum_multiplier * vol_adjustment
        
        # Cap at maximum position size
        max_position = 0.20  # Maximum 20% for momentum trades
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
            
            # Get signals using the main method
            signals = self.generate_signals(temp_data)
            
            if signals:
                signal = signals[0]
                # Extract reason from metadata or create a descriptive reason
                metadata = signal.metadata or {}
                strategy_name = metadata.get('strategy', 'momentum')
                reason = f"{strategy_name} signal, strength: {signal.strength:.2f}"
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            else:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No signal'}
                
        except Exception as e:
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
