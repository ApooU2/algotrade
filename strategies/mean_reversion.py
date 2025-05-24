"""
Mean Reversion Strategy
Based on the principle that prices tend to revert to their mean over time
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from scipy import stats

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_volume_ratio': 1.2,
            'z_score_threshold': 2,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.08
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Mean Reversion", default_params)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate mean reversion signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < max(self.parameters['bb_period'], self.parameters['rsi_period']):
                continue
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Get the latest data point
            latest = df.iloc[-1]
            recent = df.tail(5)  # Last 5 periods for trend analysis
            
            # Mean reversion conditions
            signal = self._check_mean_reversion_conditions(symbol, latest, recent, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required indicators
        """
        # Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']
        
        df['BB_middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_rolling_std = df['Close'].rolling(window=bb_period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_rolling_std * bb_std)
        df['BB_lower'] = df['BB_middle'] - (bb_rolling_std * bb_std)
        
        # Bollinger Band position (0 = lower band, 1 = upper band)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], self.parameters['rsi_period'])
        
        # Z-score of price relative to recent mean
        df['Price_zscore'] = stats.zscore(df['Close'].rolling(window=20))
        
        # Volume ratio
        df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Price momentum
        df['Price_momentum'] = df['Close'].pct_change(5)  # 5-period momentum
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI indicator
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _check_mean_reversion_conditions(self, symbol: str, latest: pd.Series, 
                                       recent: pd.DataFrame, df: pd.DataFrame) -> Signal:
        """
        Check for mean reversion entry conditions
        """
        # Oversold conditions (Buy signal)
        oversold_conditions = [
            latest['BB_position'] < 0.1,  # Price near lower Bollinger Band
            latest['RSI'] < self.parameters['rsi_oversold'],  # RSI oversold
            latest['Price_zscore'] < -self.parameters['z_score_threshold'],  # Price below normal
            latest['Volume_ratio'] > self.parameters['min_volume_ratio'],  # Higher than average volume
            recent['Close'].iloc[-1] > recent['Close'].iloc[-3]  # Recent price uptick
        ]
        
        # Overbought conditions (Sell signal)
        overbought_conditions = [
            latest['BB_position'] > 0.9,  # Price near upper Bollinger Band
            latest['RSI'] > self.parameters['rsi_overbought'],  # RSI overbought
            latest['Price_zscore'] > self.parameters['z_score_threshold'],  # Price above normal
            latest['Volume_ratio'] > self.parameters['min_volume_ratio'],  # Higher than average volume
            recent['Close'].iloc[-1] < recent['Close'].iloc[-3]  # Recent price downtick
        ]
        
        # Calculate signal strength based on how many conditions are met
        buy_strength = sum(oversold_conditions) / len(oversold_conditions)
        sell_strength = sum(overbought_conditions) / len(overbought_conditions)
        
        # Additional strength factors
        volatility_factor = min(latest.get('ATR', 0) / latest['Close'] * 100, 1.0)  # Higher volatility = higher opportunity
        
        if buy_strength >= 0.6:  # At least 60% of conditions met
            strength = buy_strength * (1 + volatility_factor * 0.2)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=min(strength, 1.0),
                price=latest['Close'],
                timestamp=latest.name,
                metadata={
                    'strategy': 'mean_reversion',
                    'bb_position': latest['BB_position'],
                    'rsi': latest['RSI'],
                    'z_score': latest['Price_zscore'],
                    'volume_ratio': latest['Volume_ratio']
                }
            )
        
        elif sell_strength >= 0.6:  # At least 60% of conditions met
            strength = sell_strength * (1 + volatility_factor * 0.2)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=min(strength, 1.0),
                price=latest['Close'],
                timestamp=latest.name,
                metadata={
                    'strategy': 'mean_reversion',
                    'bb_position': latest['BB_position'],
                    'rsi': latest['RSI'],
                    'z_score': latest['Price_zscore'],
                    'volume_ratio': latest['Volume_ratio']
                }
            )
        
        return None
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size using Kelly Criterion with modifications
        """
        # Base position size (percentage of portfolio)
        base_size = 0.05  # 5% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on volatility (lower vol = larger position)
        vol_adjustment = max(0.5, min(2.0, 0.2 / volatility))
        
        # Adjust based on RSI extreme levels
        rsi = signal.metadata.get('rsi', 50)
        if signal.signal_type == SignalType.BUY and rsi < 25:
            rsi_multiplier = 1.5  # Increase size for extreme oversold
        elif signal.signal_type == SignalType.SELL and rsi > 75:
            rsi_multiplier = 1.5  # Increase size for extreme overbought
        else:
            rsi_multiplier = 1.0
        
        # Calculate final position size
        position_size = base_size * strength_multiplier * vol_adjustment * rsi_multiplier
        
        # Cap at maximum position size
        max_position = 0.15  # Maximum 15% of portfolio
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
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': signal.description
                }
            else:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No signal'}
                
        except Exception as e:
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
