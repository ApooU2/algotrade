"""
Pairs Trading Strategy
Statistical arbitrage strategy that trades pairs of correlated stocks
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from scipy import stats
from itertools import combinations

class PairsTradingStrategy(BaseStrategy):
    """
    Pairs trading strategy based on mean reversion of price spreads
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 60,
            'min_correlation': 0.7,
            'max_correlation': 0.95,
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 3.5,
            'min_half_life': 5,
            'max_half_life': 30,
            'cointegration_pvalue': 0.05,
            'min_observations': 120,
            'hedge_ratio_method': 'ols',  # 'ols' or 'tls'
            'rebalance_frequency': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Pairs Trading", default_params)
        
        self.pairs = []
        self.spreads = {}
        self.hedge_ratios = {}
        self.last_rebalance = {}
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate pairs trading signals
        """
        signals = []
        
        # Update pairs if needed
        self._update_pairs(data)
        
        # Generate signals for each pair
        for pair in self.pairs:
            pair_signals = self._generate_pair_signals(pair, data)
            signals.extend(pair_signals)
        
        return signals
    
    def _update_pairs(self, data: Dict[str, pd.DataFrame]):
        """
        Update the list of tradeable pairs
        """
        symbols = list(data.keys())
        
        if not self.pairs or self._should_rebalance():
            print("Updating pairs...")
            self.pairs = self._find_cointegrated_pairs(data, symbols)
            print(f"Found {len(self.pairs)} tradeable pairs")
    
    def _should_rebalance(self) -> bool:
        """
        Check if pairs should be rebalanced
        """
        if not self.last_rebalance:
            return True
        
        days_since_rebalance = (pd.Timestamp.now() - self.last_rebalance.get('date', pd.Timestamp.now())).days
        return days_since_rebalance >= self.parameters['rebalance_frequency']
    
    def _find_cointegrated_pairs(self, data: Dict[str, pd.DataFrame], symbols: List[str]) -> List[Tuple[str, str]]:
        """
        Find cointegrated pairs using statistical tests
        """
        valid_pairs = []
        
        # Get aligned price data
        price_data = self._align_price_data(data, symbols)
        
        if price_data.empty or len(price_data.columns) < 2:
            return valid_pairs
        
        # Test all possible pairs
        for symbol1, symbol2 in combinations(symbols, 2):
            if symbol1 not in price_data.columns or symbol2 not in price_data.columns:
                continue
            
            try:
                pair_data = price_data[[symbol1, symbol2]].dropna()
                
                if len(pair_data) < self.parameters['min_observations']:
                    continue
                
                # Test correlation
                correlation = pair_data[symbol1].corr(pair_data[symbol2])
                
                if not (self.parameters['min_correlation'] <= abs(correlation) <= self.parameters['max_correlation']):
                    continue
                
                # Test cointegration
                is_cointegrated, pvalue, hedge_ratio = self._test_cointegration(
                    pair_data[symbol1], pair_data[symbol2]
                )
                
                if not is_cointegrated:
                    continue
                
                # Test spread mean reversion
                spread = self._calculate_spread(pair_data[symbol1], pair_data[symbol2], hedge_ratio)
                half_life = self._calculate_half_life(spread)
                
                if not (self.parameters['min_half_life'] <= half_life <= self.parameters['max_half_life']):
                    continue
                
                # Store valid pair
                pair_key = tuple(sorted([symbol1, symbol2]))
                valid_pairs.append(pair_key)
                self.hedge_ratios[pair_key] = hedge_ratio
                
                print(f"Valid pair: {symbol1}-{symbol2}, correlation: {correlation:.3f}, "
                      f"p-value: {pvalue:.3f}, half-life: {half_life:.1f}")
                
            except Exception as e:
                print(f"Error testing pair {symbol1}-{symbol2}: {e}")
                continue
        
        self.last_rebalance['date'] = pd.Timestamp.now()
        return valid_pairs
    
    def _align_price_data(self, data: Dict[str, pd.DataFrame], symbols: List[str]) -> pd.DataFrame:
        """
        Align price data for all symbols
        """
        price_data = pd.DataFrame()
        
        for symbol in symbols:
            if symbol in data and 'Close' in data[symbol].columns:
                price_data[symbol] = data[symbol]['Close']
        
        return price_data.dropna()
    
    def _test_cointegration(self, series1: pd.Series, series2: pd.Series) -> Tuple[bool, float, float]:
        """
        Test for cointegration between two price series
        """
        try:
            from statsmodels.tsa.stattools import coint
            
            # Perform cointegration test
            score, pvalue, _ = coint(series1, series2)
            
            # Calculate hedge ratio using OLS
            hedge_ratio = self._calculate_hedge_ratio(series1, series2)
            
            is_cointegrated = pvalue < self.parameters['cointegration_pvalue']
            
            return is_cointegrated, pvalue, hedge_ratio
            
        except ImportError:
            # Fallback to correlation-based test if statsmodels not available
            correlation = series1.corr(series2)
            hedge_ratio = self._calculate_hedge_ratio(series1, series2)
            
            # Use correlation as proxy (not as robust as cointegration test)
            is_cointegrated = abs(correlation) >= 0.8
            pvalue = 1 - abs(correlation)  # Approximate p-value
            
            return is_cointegrated, pvalue, hedge_ratio
    
    def _calculate_hedge_ratio(self, series1: pd.Series, series2: pd.Series) -> float:
        """
        Calculate hedge ratio between two series
        """
        if self.parameters['hedge_ratio_method'] == 'ols':
            # Ordinary Least Squares
            slope, _, _, _, _ = stats.linregress(series2, series1)
            return slope
        else:
            # Total Least Squares (orthogonal regression)
            # Simplified implementation
            return series1.cov(series2) / series2.var()
    
    def _calculate_spread(self, series1: pd.Series, series2: pd.Series, hedge_ratio: float) -> pd.Series:
        """
        Calculate spread between two series
        """
        return series1 - hedge_ratio * series2
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion
        """
        try:
            # Use AR(1) model to estimate half-life
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align series
            common_index = spread_lag.index.intersection(spread_diff.index)
            spread_lag = spread_lag[common_index]
            spread_diff = spread_diff[common_index]
            
            if len(spread_lag) < 10:
                return float('inf')
            
            # Linear regression
            slope, _, _, _, _ = stats.linregress(spread_lag, spread_diff)
            
            # Half-life calculation
            if slope >= 0:
                return float('inf')  # No mean reversion
            
            half_life = -np.log(2) / slope
            
            return max(half_life, 1)  # Minimum 1 day
            
        except:
            return float('inf')
    
    def _generate_pair_signals(self, pair: Tuple[str, str], data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate signals for a specific pair
        """
        signals = []
        symbol1, symbol2 = pair
        
        if symbol1 not in data or symbol2 not in data:
            return signals
        
        try:
            # Get price data
            prices1 = data[symbol1]['Close']
            prices2 = data[symbol2]['Close']
            
            # Align data
            common_index = prices1.index.intersection(prices2.index)
            prices1 = prices1[common_index]
            prices2 = prices2[common_index]
            
            if len(prices1) < self.parameters['lookback_period']:
                return signals
            
            # Calculate spread
            hedge_ratio = self.hedge_ratios.get(pair, 1.0)
            spread = self._calculate_spread(prices1, prices2, hedge_ratio)
            
            # Calculate z-score
            lookback = self.parameters['lookback_period']
            spread_mean = spread.rolling(window=lookback).mean()
            spread_std = spread.rolling(window=lookback).std()
            z_score = (spread - spread_mean) / spread_std
            
            # Get latest values
            latest_z_score = z_score.iloc[-1]
            latest_spread = spread.iloc[-1]
            
            # Generate signals based on z-score
            if abs(latest_z_score) >= self.parameters['entry_zscore']:
                
                if latest_z_score > 0:  # Spread too high, short spread
                    # Short symbol1, long symbol2
                    signal1 = Signal(
                        symbol=symbol1,
                        signal_type=SignalType.SELL,
                        strength=min(abs(latest_z_score) / self.parameters['entry_zscore'], 1.0),
                        price=prices1.iloc[-1],
                        timestamp=prices1.index[-1],
                        metadata={
                            'strategy': 'pairs_trading',
                            'pair': pair,
                            'z_score': latest_z_score,
                            'spread': latest_spread,
                            'hedge_ratio': hedge_ratio,
                            'position_type': 'short_spread'
                        }
                    )
                    
                    signal2 = Signal(
                        symbol=symbol2,
                        signal_type=SignalType.BUY,
                        strength=min(abs(latest_z_score) / self.parameters['entry_zscore'], 1.0),
                        price=prices2.iloc[-1],
                        timestamp=prices2.index[-1],
                        metadata={
                            'strategy': 'pairs_trading',
                            'pair': pair,
                            'z_score': latest_z_score,
                            'spread': latest_spread,
                            'hedge_ratio': hedge_ratio,
                            'position_type': 'long_spread'
                        }
                    )
                    
                else:  # Spread too low, long spread
                    # Long symbol1, short symbol2
                    signal1 = Signal(
                        symbol=symbol1,
                        signal_type=SignalType.BUY,
                        strength=min(abs(latest_z_score) / self.parameters['entry_zscore'], 1.0),
                        price=prices1.iloc[-1],
                        timestamp=prices1.index[-1],
                        metadata={
                            'strategy': 'pairs_trading',
                            'pair': pair,
                            'z_score': latest_z_score,
                            'spread': latest_spread,
                            'hedge_ratio': hedge_ratio,
                            'position_type': 'long_spread'
                        }
                    )
                    
                    signal2 = Signal(
                        symbol=symbol2,
                        signal_type=SignalType.SELL,
                        strength=min(abs(latest_z_score) / self.parameters['entry_zscore'], 1.0),
                        price=prices2.iloc[-1],
                        timestamp=prices2.index[-1],
                        metadata={
                            'strategy': 'pairs_trading',
                            'pair': pair,
                            'z_score': latest_z_score,
                            'spread': latest_spread,
                            'hedge_ratio': hedge_ratio,
                            'position_type': 'short_spread'
                        }
                    )
                
                signals.extend([signal1, signal2])
        
        except Exception as e:
            print(f"Error generating signals for pair {pair}: {e}")
        
        return signals
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for pairs trading
        """
        # Base position size (pairs trading typically uses smaller positions)
        base_size = 0.04  # 4% base allocation per leg
        
        # Adjust based on signal strength (z-score magnitude)
        strength_multiplier = signal.strength
        
        # Adjust based on hedge ratio (for proper hedging)
        hedge_ratio = signal.metadata.get('hedge_ratio', 1.0)
        
        # Calculate position size
        if signal.symbol == signal.metadata['pair'][0]:
            # First symbol in pair
            position_size = base_size * strength_multiplier
        else:
            # Second symbol in pair (adjust by hedge ratio)
            position_size = base_size * strength_multiplier * hedge_ratio
        
        # Cap at maximum position size
        max_position = 0.08  # Maximum 8% per leg
        position_size = min(position_size, max_position)
        
        return position_size * portfolio_value / signal.price
