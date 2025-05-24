"""
Market Microstructure Strategy - Analyzes order flow, bid-ask spreads, and market depth
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class MarketMicrostructureStrategy(BaseStrategy):
    """
    Market Microstructure strategy that analyzes order flow imbalances,
    bid-ask spreads, and market depth indicators for trading signals
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'spread_threshold': 0.005,      # Max bid-ask spread percentage
            'volume_imbalance_threshold': 0.6, # Order flow imbalance threshold
            'price_impact_period': 10,      # Period for price impact analysis
            'tick_size_analysis': True,     # Analyze tick-by-tick movements
            'market_depth_levels': 5,       # Number of depth levels to analyze
            'aggressive_order_threshold': 0.7, # Threshold for aggressive orders
            'liquidity_threshold': 100000,  # Minimum liquidity requirement
            'microstructure_period': 20,    # Period for microstructure analysis
            'order_flow_window': 15,        # Window for order flow calculation
            'price_efficiency_threshold': 0.8, # Market efficiency threshold
            'stop_loss_atr_mult': 1.5,
            'take_profit_ratio': 2.0,
            'lookback_period': 50
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Market Microstructure", default_params)
        
        # Store microstructure data
        self.order_flow_data = {}
        self.spread_history = {}
        self.liquidity_metrics = {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate Market Microstructure signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['lookback_period']:
                continue
                
            try:
                # Calculate microstructure indicators
                df_indicators = self._calculate_microstructure_indicators(df.copy())
                
                # Analyze order flow patterns
                self._analyze_order_flow(symbol, df_indicators)
                
                # Generate signals based on microstructure analysis
                symbol_signals = self._generate_microstructure_signals(symbol, df_indicators)
                signals.extend(symbol_signals)
                
            except Exception as e:
                print(f"Error generating microstructure signals for {symbol}: {e}")
                continue
                
        return signals
    
    def _calculate_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market microstructure indicators
        """
        # Simulated bid-ask spread (in real trading, this would come from L2 data)
        df['Spread'] = self._estimate_bid_ask_spread(df)
        df['Spread_Pct'] = df['Spread'] / df['Close']
        
        # Volume-weighted price movements
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Price impact estimation
        df['Price_Impact'] = self._calculate_price_impact(df)
        
        # Order flow imbalance estimation
        df['Order_Flow_Imbalance'] = self._estimate_order_flow_imbalance(df)
        
        # Tick direction analysis
        df['Tick_Direction'] = self._calculate_tick_direction(df)
        df['Tick_Direction_MA'] = df['Tick_Direction'].rolling(window=10).mean()
        
        # Liquidity estimation
        df['Liquidity_Score'] = self._estimate_liquidity(df)
        
        # Market efficiency metrics
        df['Price_Efficiency'] = self._calculate_price_efficiency(df)
        
        # Aggressive vs passive order estimation
        df['Aggressive_Orders'] = self._estimate_aggressive_orders(df)
        
        # Average True Range
        df['ATR'] = self._calculate_atr(df, 14)
        
        # Relative Volume
        df['Rel_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        return df
    
    def _estimate_bid_ask_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate bid-ask spread using high-low range and volume
        """
        # Simple estimation: use fraction of high-low range adjusted by volume
        hl_range = df['High'] - df['Low']
        volume_factor = 1 / (1 + np.log(df['Volume'] / df['Volume'].rolling(50).mean()))
        
        # Higher volume typically means tighter spreads
        estimated_spread = hl_range * volume_factor * 0.1  # Scale factor
        return estimated_spread.fillna(0)
    
    def _calculate_price_impact(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate price impact of trades
        """
        # Price impact = (Close - Open) / Volume^0.5
        price_change = df['Close'] - df['Open']
        volume_sqrt = np.sqrt(df['Volume'])
        
        price_impact = price_change / volume_sqrt * 1000000  # Scale for readability
        return price_impact.fillna(0)
    
    def _estimate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate order flow imbalance (buy vs sell pressure)
        """
        # Use price and volume to estimate buy/sell imbalance
        price_change = df['Close'].diff()
        volume = df['Volume']
        
        # Positive price change with high volume suggests buy pressure
        buy_volume = np.where(price_change > 0, volume, 0)
        sell_volume = np.where(price_change < 0, volume, 0)
        
        # Calculate rolling imbalance
        window = self.parameters['order_flow_window']
        total_buy = pd.Series(buy_volume).rolling(window=window).sum()
        total_sell = pd.Series(sell_volume).rolling(window=window).sum()
        
        imbalance = (total_buy - total_sell) / (total_buy + total_sell + 1e-10)
        return imbalance.fillna(0)
    
    def _calculate_tick_direction(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate tick direction (uptick/downtick)
        """
        price_diff = df['Close'].diff()
        tick_direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
        return pd.Series(tick_direction, index=df.index)
    
    def _estimate_liquidity(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate market liquidity
        """
        # Combine volume and spread for liquidity estimation
        volume_score = df['Volume'] / df['Volume'].rolling(50).mean()
        spread_score = 1 / (1 + df['Spread_Pct'] * 100)  # Lower spread = higher liquidity
        
        liquidity_score = (volume_score * 0.7) + (spread_score * 0.3)
        return liquidity_score.fillna(0.5)
    
    def _calculate_price_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate price efficiency (how quickly prices adjust to new information)
        """
        # Use autocorrelation of returns as efficiency measure
        returns = df['Close'].pct_change()
        
        efficiency_scores = []
        window = self.parameters['microstructure_period']
        
        for i in range(len(returns)):
            if i >= window:
                recent_returns = returns.iloc[i-window:i]
                # Lower autocorrelation = higher efficiency
                autocorr = recent_returns.autocorr(lag=1)
                efficiency = 1 - abs(autocorr) if not np.isnan(autocorr) else 0.5
                efficiency_scores.append(max(0, min(1, efficiency)))
            else:
                efficiency_scores.append(0.5)
        
        return pd.Series(efficiency_scores, index=df.index)
    
    def _estimate_aggressive_orders(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate percentage of aggressive (market) orders vs passive (limit) orders
        """
        # High volume with price movement suggests aggressive orders
        price_movement = abs(df['Close'] - df['Open']) / df['Open']
        volume_intensity = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # More price movement with volume suggests more aggressive orders
        aggressive_ratio = (price_movement * volume_intensity).rolling(10).mean()
        return aggressive_ratio.fillna(0.5)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _analyze_order_flow(self, symbol: str, df: pd.DataFrame):
        """
        Analyze order flow patterns for the symbol
        """
        recent_df = df.tail(self.parameters['microstructure_period'])
        
        # Store order flow metrics
        self.order_flow_data[symbol] = {
            'avg_imbalance': recent_df['Order_Flow_Imbalance'].mean(),
            'imbalance_std': recent_df['Order_Flow_Imbalance'].std(),
            'avg_aggressive_ratio': recent_df['Aggressive_Orders'].mean(),
            'avg_liquidity': recent_df['Liquidity_Score'].mean(),
            'avg_efficiency': recent_df['Price_Efficiency'].mean(),
            'timestamp': df.index[-1]
        }
        
        # Store spread history
        self.spread_history[symbol] = {
            'avg_spread_pct': recent_df['Spread_Pct'].mean(),
            'spread_volatility': recent_df['Spread_Pct'].std(),
            'current_spread': recent_df['Spread_Pct'].iloc[-1]
        }
    
    def _generate_microstructure_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on microstructure analysis
        """
        signals = []
        
        for i in range(len(df) - 10, len(df)):
            if i < 30:
                continue
                
            current = df.iloc[i]
            recent_df = df.iloc[max(0, i-10):i+1]
            
            signal = self._analyze_microstructure_patterns(symbol, current, recent_df, i, df)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_microstructure_patterns(self, symbol: str, current: pd.Series, 
                                       recent_df: pd.DataFrame, index: int, 
                                       full_df: pd.DataFrame) -> Signal:
        """
        Analyze microstructure patterns for signal generation
        """
        current_price = current['Close']
        
        signal_type = None
        signal_strength = 0
        reasons = []
        
        # Check spread conditions
        if current['Spread_Pct'] > self.parameters['spread_threshold']:
            # Spread too wide, skip trading
            return None
        
        # Check liquidity
        if current['Liquidity_Score'] < 0.5:
            # Low liquidity, reduce confidence
            signal_strength -= 0.2
        
        # Order flow imbalance signal
        order_flow = current['Order_Flow_Imbalance']
        if order_flow > self.parameters['volume_imbalance_threshold']:
            signal_type = SignalType.BUY
            signal_strength += 0.4
            reasons.append("buy_flow_imbalance")
        elif order_flow < -self.parameters['volume_imbalance_threshold']:
            signal_type = SignalType.SELL
            signal_strength += 0.4
            reasons.append("sell_flow_imbalance")
        
        # Aggressive order flow signal
        if current['Aggressive_Orders'] > self.parameters['aggressive_order_threshold']:
            if current['Close'] > current['Open']:  # Bullish aggressive flow
                if signal_type is None or signal_type == SignalType.BUY:
                    signal_type = SignalType.BUY
                    signal_strength += 0.3
                    reasons.append("aggressive_buying")
            else:  # Bearish aggressive flow
                if signal_type is None or signal_type == SignalType.SELL:
                    signal_type = SignalType.SELL
                    signal_strength += 0.3
                    reasons.append("aggressive_selling")
        
        # Price efficiency signal
        if current['Price_Efficiency'] < self.parameters['price_efficiency_threshold']:
            # Inefficient pricing may create opportunities
            tick_direction = recent_df['Tick_Direction_MA'].iloc[-1]
            if tick_direction > 0.3 and (signal_type is None or signal_type == SignalType.BUY):
                signal_type = SignalType.BUY
                signal_strength += 0.2
                reasons.append("inefficiency_uptick")
            elif tick_direction < -0.3 and (signal_type is None or signal_type == SignalType.SELL):
                signal_type = SignalType.SELL
                signal_strength += 0.2
                reasons.append("inefficiency_downtick")
        
        # Price impact confirmation
        if abs(current['Price_Impact']) > recent_df['Price_Impact'].std() * 2:
            if current['Price_Impact'] > 0 and signal_type == SignalType.BUY:
                signal_strength += 0.15
                reasons.append("high_buy_impact")
            elif current['Price_Impact'] < 0 and signal_type == SignalType.SELL:
                signal_strength += 0.15
                reasons.append("high_sell_impact")
        
        # VWAP relative position
        if current_price > current['VWAP'] and signal_type == SignalType.BUY:
            signal_strength += 0.1
            reasons.append("above_vwap")
        elif current_price < current['VWAP'] and signal_type == SignalType.SELL:
            signal_strength += 0.1
            reasons.append("below_vwap")
        
        # Volume confirmation
        if current['Rel_Volume'] > 1.5:
            signal_strength += 0.1
            reasons.append("high_volume")
        
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
                'strategy': 'market_microstructure',
                'reasons': reasons,
                'order_flow_imbalance': order_flow,
                'aggressive_orders': current['Aggressive_Orders'],
                'liquidity_score': current['Liquidity_Score'],
                'price_efficiency': current['Price_Efficiency'],
                'spread_pct': current['Spread_Pct'],
                'price_impact': current['Price_Impact'],
                'rel_volume': current['Rel_Volume'],
                'atr': current['ATR']
            }
        )
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for microstructure trades
        """
        # Base position size (microstructure trades are typically smaller)
        base_size = 0.06  # 6% base allocation
        
        # Adjust based on signal strength
        strength_multiplier = signal.strength
        
        # Adjust based on liquidity
        liquidity_score = signal.metadata.get('liquidity_score', 0.5)
        liquidity_multiplier = 0.5 + liquidity_score  # 0.5 to 1.5 range
        
        # Adjust based on spread
        spread_pct = signal.metadata.get('spread_pct', 0.01)
        spread_multiplier = max(0.5, 1 - (spread_pct / 0.01))  # Reduce size for wide spreads
        
        # Adjust based on order flow strength
        order_flow = abs(signal.metadata.get('order_flow_imbalance', 0))
        flow_multiplier = 1 + min(order_flow * 0.5, 0.3)  # Up to 30% increase
        
        # Volatility adjustment
        vol_multiplier = max(0.5, min(1.2, 1 / (1 + volatility * 3)))
        
        # Calculate final position size
        position_size = (base_size * strength_multiplier * liquidity_multiplier * 
                        spread_multiplier * flow_multiplier * vol_multiplier)
        
        # Cap at maximum position size
        max_position = 0.10  # Maximum 10% per position
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
                order_flow = metadata.get('order_flow_imbalance', 0)
                
                reason = f"microstructure: {', '.join(reasons)}, flow: {order_flow:.2f}"
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            else:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No microstructure signal'}
                
        except Exception as e:
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
