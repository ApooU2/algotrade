"""
Gap Trading Strategy - Identifies and trades various types of price gaps
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class GapTradingStrategy(BaseStrategy):
    """
    Gap Trading strategy that identifies different types of price gaps
    and trades gap fills, continuations, and breakaway patterns
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'min_gap_percent': 1.0,        # Minimum gap size (1%)
            'max_gap_percent': 8.0,        # Maximum gap size (8%)
            'volume_confirmation': True,    # Require volume confirmation
            'min_volume_ratio': 1.5,       # Minimum volume ratio for confirmation
            'gap_fill_threshold': 0.75,    # Threshold for gap fill trades (75%)
            'continuation_threshold': 0.25, # Threshold for continuation trades (25%)
            'overnight_gap_only': True,    # Only trade overnight gaps
            'premarket_analysis': True,    # Analyze pre-market activity
            'sector_correlation': True,    # Check sector/market correlation
            'news_sentiment_weight': 0.3,  # Weight for news sentiment (if available)
            'risk_reward_ratio': 2.0,      # Minimum risk/reward ratio
            'stop_loss_gap_percent': 0.5,  # Stop loss as % of gap size
            'lookback_period': 60
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Gap Trading", default_params)
        
        # Store gap data
        self.gaps_history = {}
        self.gap_statistics = {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate Gap Trading signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['lookback_period']:
                continue
                
            try:
                # Calculate gap indicators and identify gaps
                df_indicators = self._calculate_gap_indicators(df.copy())
                
                # Identify and classify gaps
                self._identify_gaps(symbol, df_indicators)
                
                # Generate signals based on gap analysis
                symbol_signals = self._generate_gap_signals(symbol, df_indicators)
                signals.extend(symbol_signals)
                
            except Exception as e:
                print(f"Error generating gap signals for {symbol}: {e}")
                continue
                
        return signals
    
    def _calculate_gap_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate gap-related indicators
        """
        # Calculate gaps
        df['Prev_Close'] = df['Close'].shift(1)
        df['Gap_Size'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close'] * 100
        df['Gap_Direction'] = np.where(df['Gap_Size'] > 0, 1, np.where(df['Gap_Size'] < 0, -1, 0))
        
        # Gap classification
        df['Is_Gap'] = abs(df['Gap_Size']) >= self.parameters['min_gap_percent']
        df['Gap_Type'] = self._classify_gap_type(df)
        
        # Volume analysis
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # Price action after gap
        df['Gap_Fill_Percent'] = self._calculate_gap_fill_percent(df)
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open'] * 100
        
        # Volatility measures
        df['ATR'] = self._calculate_atr(df, 14)
        df['Volatility_Ratio'] = df['ATR'] / df['Close']
        
        # Trend context
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Trend_Short'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
        df['Trend_Long'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        
        # Support and resistance levels
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        # Momentum indicators
        df['RSI'] = self._calculate_rsi(df, 14)
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df)
        
        return df
    
    def _classify_gap_type(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify gap types: breakaway, runaway, exhaustion, or common
        """
        gap_types = []
        
        for i in range(len(df)):
            if not df['Is_Gap'].iloc[i]:
                gap_types.append('none')
                continue
            
            # Look at context around the gap
            start_idx = max(0, i - 20)
            end_idx = min(len(df), i + 5)
            context = df.iloc[start_idx:end_idx]
            
            gap_size = abs(df['Gap_Size'].iloc[i])
            
            # Check for volume pattern
            avg_volume = context['Volume'].mean()
            current_volume = df['Volume'].iloc[i]
            
            # Check for price pattern before gap
            pre_gap_volatility = context['ATR'].iloc[-5:].mean() if len(context) >= 5 else 0
            
            # Classify based on patterns
            if gap_size > 3.0 and current_volume > avg_volume * 2:
                if i > 20:
                    # Check if breaking out of consolidation
                    recent_range = context['High'].max() - context['Low'].min()
                    if recent_range / context['Close'].mean() < 0.1:  # Tight consolidation
                        gap_types.append('breakaway')
                    else:
                        gap_types.append('runaway')
                else:
                    gap_types.append('breakaway')
            elif gap_size > 5.0:
                gap_types.append('exhaustion')
            else:
                gap_types.append('common')
        
        return pd.Series(gap_types, index=df.index)
    
    def _calculate_gap_fill_percent(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate what percentage of the gap has been filled during the day
        """
        gap_fill_percent = []
        
        for i in range(len(df)):
            if not df['Is_Gap'].iloc[i]:
                gap_fill_percent.append(0)
                continue
            
            gap_size = df['Gap_Size'].iloc[i]
            prev_close = df['Prev_Close'].iloc[i]
            current_open = df['Open'].iloc[i]
            current_close = df['Close'].iloc[i]
            
            if gap_size > 0:  # Gap up
                # Check how much of the gap has been filled (price moving back toward prev_close)
                if current_close < current_open:
                    fill_amount = current_open - current_close
                    gap_amount = current_open - prev_close
                    fill_percent = (fill_amount / gap_amount) * 100
                else:
                    fill_percent = 0
            else:  # Gap down
                # Check how much of the gap has been filled (price moving back toward prev_close)
                if current_close > current_open:
                    fill_amount = current_close - current_open
                    gap_amount = prev_close - current_open
                    fill_percent = (fill_amount / gap_amount) * 100
                else:
                    fill_percent = 0
            
            gap_fill_percent.append(min(100, max(0, fill_percent)))
        
        return pd.Series(gap_fill_percent, index=df.index)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _identify_gaps(self, symbol: str, df: pd.DataFrame):
        """
        Identify and store gap information for the symbol
        """
        recent_gaps = df[df['Is_Gap']].tail(10)  # Last 10 gaps
        
        if len(recent_gaps) == 0:
            return
        
        gap_info = []
        for idx, row in recent_gaps.iterrows():
            gap_info.append({
                'date': idx,
                'gap_size': row['Gap_Size'],
                'gap_type': row['Gap_Type'],
                'volume_ratio': row['Volume_Ratio'],
                'fill_percent': row['Gap_Fill_Percent'],
                'direction': row['Gap_Direction']
            })
        
        self.gaps_history[symbol] = gap_info
        
        # Calculate gap statistics
        all_gaps = df[df['Is_Gap']]
        if len(all_gaps) > 0:
            self.gap_statistics[symbol] = {
                'avg_gap_size': all_gaps['Gap_Size'].abs().mean(),
                'gap_fill_rate': (all_gaps['Gap_Fill_Percent'] > 50).mean(),
                'avg_fill_percent': all_gaps['Gap_Fill_Percent'].mean(),
                'gap_frequency': len(all_gaps) / len(df) * 100,
                'up_gap_rate': (all_gaps['Gap_Direction'] == 1).mean()
            }
    
    def _generate_gap_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on gap analysis
        """
        signals = []
        
        # Only check the last few days for new gaps
        for i in range(len(df) - 5, len(df)):
            if i < 20:
                continue
                
            current = df.iloc[i]
            
            # Only trade significant gaps
            if not current['Is_Gap'] or abs(current['Gap_Size']) < self.parameters['min_gap_percent']:
                continue
            
            # Skip gaps that are too large (likely news-driven)
            if abs(current['Gap_Size']) > self.parameters['max_gap_percent']:
                continue
            
            signal = self._analyze_gap_patterns(symbol, current, df, i)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_gap_patterns(self, symbol: str, current: pd.Series, 
                            df: pd.DataFrame, index: int) -> Signal:
        """
        Analyze gap patterns for signal generation
        """
        gap_size = current['Gap_Size']
        gap_type = current['Gap_Type']
        
        signal_type = None
        signal_strength = 0
        reasons = []
        
        # Volume confirmation
        if (self.parameters['volume_confirmation'] and 
            current['Volume_Ratio'] < self.parameters['min_volume_ratio']):
            return None
        
        # Gap fill strategy
        if current['Gap_Fill_Percent'] < self.parameters['gap_fill_threshold']:
            if gap_size > 0:  # Gap up, expect fill (sell signal)
                signal_type = SignalType.SELL
                signal_strength += 0.4
                reasons.append("gap_up_fill_expected")
            else:  # Gap down, expect fill (buy signal)
                signal_type = SignalType.BUY
                signal_strength += 0.4
                reasons.append("gap_down_fill_expected")
        
        # Gap continuation strategy (for breakaway/runaway gaps)
        elif (gap_type in ['breakaway', 'runaway'] and 
              current['Gap_Fill_Percent'] < self.parameters['continuation_threshold']):
            if gap_size > 0:  # Gap up, expect continuation (buy signal)
                signal_type = SignalType.BUY
                signal_strength += 0.5
                reasons.append("gap_up_continuation")
            else:  # Gap down, expect continuation (sell signal)
                signal_type = SignalType.SELL
                signal_strength += 0.5
                reasons.append("gap_down_continuation")
        
        # Additional confirmations
        
        # Trend alignment
        if signal_type == SignalType.BUY and current['Trend_Short'] == 1:
            signal_strength += 0.15
            reasons.append("trend_aligned_bullish")
        elif signal_type == SignalType.SELL and current['Trend_Short'] == -1:
            signal_strength += 0.15
            reasons.append("trend_aligned_bearish")
        
        # Volume confirmation
        if current['Volume_Ratio'] > 2.0:
            signal_strength += 0.15
            reasons.append("high_volume_confirmation")
        
        # RSI levels
        if signal_type == SignalType.BUY and current['RSI'] < 40:
            signal_strength += 0.1
            reasons.append("rsi_oversold")
        elif signal_type == SignalType.SELL and current['RSI'] > 60:
            signal_strength += 0.1
            reasons.append("rsi_overbought")
        
        # MACD confirmation
        if signal_type == SignalType.BUY and current['MACD'] > current['MACD_Signal']:
            signal_strength += 0.1
            reasons.append("macd_bullish")
        elif signal_type == SignalType.SELL and current['MACD'] < current['MACD_Signal']:
            signal_strength += 0.1
            reasons.append("macd_bearish")
        
        # Gap size scoring
        if self.parameters['min_gap_percent'] <= abs(gap_size) <= 3.0:
            signal_strength += 0.1  # Moderate gaps are more reliable
            reasons.append("optimal_gap_size")
        
        # Historical gap statistics
        if symbol in self.gap_statistics:
            stats = self.gap_statistics[symbol]
            if stats['gap_fill_rate'] > 0.7 and "fill_expected" in reasons[0]:
                signal_strength += 0.1
                reasons.append("historical_fill_tendency")
        
        # Only generate signal if we have sufficient strength
        if signal_strength < 0.6 or signal_type is None:
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
                'strategy': 'gap_trading',
                'reasons': reasons,
                'gap_size': gap_size,
                'gap_type': gap_type,
                'gap_fill_percent': current['Gap_Fill_Percent'],
                'volume_ratio': current['Volume_Ratio'],
                'rsi': current['RSI'],
                'trend_alignment': current['Trend_Short'],
                'atr': current['ATR']
            }
        )
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for gap trades
        """
        # Base position size (gap trades can be more volatile)
        base_size = 0.07  # 7% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on gap size
        gap_size = abs(signal.metadata.get('gap_size', 0))
        if gap_size <= 2.0:
            gap_multiplier = 1.2  # Smaller gaps are safer
        elif gap_size <= 4.0:
            gap_multiplier = 1.0
        else:
            gap_multiplier = 0.8  # Larger gaps are riskier
        
        # Adjust based on gap type
        gap_type = signal.metadata.get('gap_type', 'common')
        if gap_type == 'breakaway':
            type_multiplier = 1.3  # Breakaway gaps are good opportunities
        elif gap_type == 'runaway':
            type_multiplier = 1.1
        else:
            type_multiplier = 1.0
        
        # Volume confirmation adjustment
        volume_ratio = signal.metadata.get('volume_ratio', 1)
        volume_multiplier = 1 + min((volume_ratio - 1) * 0.1, 0.3)
        
        # Volatility adjustment (gap trades need more conservative sizing)
        vol_multiplier = max(0.4, min(1.0, 1 / (1 + volatility * 4)))
        
        # Calculate final position size
        position_size = (base_size * strength_multiplier * gap_multiplier * 
                        type_multiplier * volume_multiplier * vol_multiplier)
        
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
            
            # Get signals using the main method
            signals = self.generate_signals(temp_data)
            
            if signals:
                signal = signals[0]
                metadata = signal.metadata or {}
                reasons = metadata.get('reasons', [])
                gap_size = metadata.get('gap_size', 0)
                gap_type = metadata.get('gap_type', 'none')
                
                reason = f"gap_trading: {gap_type} gap {gap_size:.1f}%, {', '.join(reasons[:2])}"
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            else:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No gap trading signal'}
                
        except Exception as e:
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
