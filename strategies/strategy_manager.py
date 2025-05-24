"""
Strategy Manager - Orchestrates multiple trading strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import asyncio
import schedule
import time
from concurrent.futures import ThreadPoolExecutor
import logging

from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy
from strategies.ml_ensemble import MLEnsembleStrategy
from strategies.pairs_trading import PairsTradingStrategy
from data.data_manager import data_manager
from execution.execution_engine import execution_engine
from config.config import CONFIG

class StrategyManager:
    """
    Manages multiple trading strategies and combines their signals
    """
    
    def __init__(self):
        self.strategies = {}
        self.strategy_weights = {}
        self.strategy_performance = {}
        self.signals_history = []
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Performance tracking
        self.last_portfolio_value = CONFIG.INITIAL_CAPITAL
        self.performance_window = 30  # Days to track for adaptive weighting
        
    def _initialize_strategies(self):
        """
        Initialize all trading strategies
        """
        # Mean Reversion Strategy
        self.strategies['mean_reversion'] = MeanReversionStrategy({
            'bb_period': 20,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'min_volume_ratio': 1.5
        })
        
        # Momentum Strategy
        self.strategies['momentum'] = MomentumStrategy({
            'short_ma': 10,
            'long_ma': 30,
            'rsi_momentum_threshold': 60,
            'min_momentum_threshold': 0.03
        })
        
        # Breakout Strategy
        self.strategies['breakout'] = BreakoutStrategy({
            'lookback_period': 25,
            'min_volume_ratio': 2.0,
            'breakout_threshold': 0.015,
            'min_consolidation_days': 7
        })
        
        # ML Ensemble Strategy
        self.strategies['ml_ensemble'] = MLEnsembleStrategy({
            'min_prediction_confidence': 0.65,
            'retrain_frequency': 21,
            'ensemble_threshold': 0.75
        })
        
        # Pairs Trading Strategy
        self.strategies['pairs_trading'] = PairsTradingStrategy({
            'min_correlation': 0.75,
            'entry_zscore': 2.2,
            'exit_zscore': 0.3,
            'max_half_life': 25
        })
        
        # Initialize equal weights
        num_strategies = len(self.strategies)
        for strategy_name in self.strategies.keys():
            self.strategy_weights[strategy_name] = 1.0 / num_strategies
            self.strategy_performance[strategy_name] = {
                'returns': [],
                'signals_generated': 0,
                'signals_profitable': 0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'last_update': datetime.now()
            }
        
        logging.info(f"Initialized {num_strategies} strategies with equal weights")
    
    def generate_combined_signals(self, data: Dict[str, pd.DataFrame]) -> List:
        """
        Generate signals from all strategies and combine them
        """
        all_signals = []
        strategy_signals = {}
        
        # Generate signals from each strategy
        with ThreadPoolExecutor(max_workers=len(self.strategies)) as executor:
            futures = {}
            
            for name, strategy in self.strategies.items():
                future = executor.submit(self._safe_generate_signals, strategy, data, name)
                futures[future] = name
            
            for future in futures:
                strategy_name = futures[future]
                try:
                    signals = future.result(timeout=30)  # 30 second timeout
                    strategy_signals[strategy_name] = signals
                    logging.info(f"{strategy_name}: Generated {len(signals)} signals")
                except Exception as e:
                    logging.error(f"Error generating signals for {strategy_name}: {e}")
                    strategy_signals[strategy_name] = []
        
        # Combine and weight signals
        combined_signals = self._combine_signals(strategy_signals)
        
        # Filter and rank signals
        final_signals = self._filter_and_rank_signals(combined_signals, data)
        
        # Store signals for performance tracking
        self.signals_history.append({
            'timestamp': datetime.now(),
            'signals': final_signals,
            'strategy_breakdown': {name: len(sigs) for name, sigs in strategy_signals.items()}
        })
        
        logging.info(f"Final combined signals: {len(final_signals)}")
        return final_signals
    
    def _safe_generate_signals(self, strategy, data, strategy_name):
        """
        Safely generate signals with error handling
        """
        try:
            return strategy.generate_signals(data)
        except Exception as e:
            logging.error(f"Error in {strategy_name}: {e}")
            return []
    
    def _combine_signals(self, strategy_signals: Dict[str, List]) -> List:
        """
        Combine signals from multiple strategies
        """
        combined_signals = []
        signal_groups = {}  # Group signals by symbol
        
        # Group signals by symbol
        for strategy_name, signals in strategy_signals.items():
            strategy_weight = self.strategy_weights[strategy_name]
            
            for signal in signals:
                symbol = signal.symbol
                if symbol not in signal_groups:
                    signal_groups[symbol] = []
                
                # Weight the signal strength by strategy weight
                weighted_signal = signal
                weighted_signal.strength *= strategy_weight
                weighted_signal.metadata['original_strength'] = signal.strength
                weighted_signal.metadata['strategy_weight'] = strategy_weight
                
                signal_groups[symbol].append((strategy_name, weighted_signal))
        
        # Combine signals for each symbol
        for symbol, signal_list in signal_groups.items():
            combined_signal = self._merge_symbol_signals(symbol, signal_list)
            if combined_signal:
                combined_signals.append(combined_signal)
        
        return combined_signals
    
    def _merge_symbol_signals(self, symbol: str, signal_list: List) -> object:
        """
        Merge multiple signals for the same symbol
        """
        if not signal_list:
            return None
        
        # Separate buy and sell signals
        buy_signals = [(name, sig) for name, sig in signal_list if sig.signal_type.value > 0]
        sell_signals = [(name, sig) for name, sig in signal_list if sig.signal_type.value < 0]
        
        # Calculate net signal strength
        buy_strength = sum(sig.strength for _, sig in buy_signals)
        sell_strength = sum(sig.strength for _, sig in sell_signals)
        net_strength = buy_strength - sell_strength
        
        # Determine final signal direction
        if abs(net_strength) < 0.3:  # Too weak, no signal
            return None
        
        # Create combined signal
        base_signal = signal_list[0][1]  # Use first signal as base
        
        if net_strength > 0:  # Net buy
            from strategies.base_strategy import SignalType
            base_signal.signal_type = SignalType.BUY
            base_signal.strength = min(buy_strength, 1.0)
        else:  # Net sell
            from strategies.base_strategy import SignalType
            base_signal.signal_type = SignalType.SELL
            base_signal.strength = min(sell_strength, 1.0)
        
        # Combine metadata
        base_signal.metadata['combined_signal'] = True
        base_signal.metadata['contributing_strategies'] = [name for name, _ in signal_list]
        base_signal.metadata['buy_signals'] = len(buy_signals)
        base_signal.metadata['sell_signals'] = len(sell_signals)
        base_signal.metadata['consensus_strength'] = len(signal_list)
        
        return base_signal
    
    def _filter_and_rank_signals(self, signals: List, data: Dict[str, pd.DataFrame]) -> List:
        """
        Filter and rank signals based on various criteria
        """
        filtered_signals = []
        
        for signal in signals:
            # Basic filters
            if signal.strength < 0.4:  # Minimum strength threshold
                continue
            
            if signal.symbol not in data:
                continue
            
            # Check data quality
            symbol_data = data[signal.symbol]
            if len(symbol_data) < 20:  # Need sufficient data
                continue
            
            # Check liquidity (volume)
            recent_volume = symbol_data['Volume'].tail(5).mean()
            if recent_volume < 100000:  # Minimum daily volume
                continue
            
            # Check volatility (avoid extremely volatile stocks)
            recent_returns = symbol_data['Close'].pct_change().tail(20)
            volatility = recent_returns.std() * np.sqrt(252)
            if volatility > 0.8:  # Max 80% annualized volatility
                continue
            
            # Add volatility and liquidity to metadata
            signal.metadata['volatility'] = volatility
            signal.metadata['avg_volume'] = recent_volume
            
            filtered_signals.append(signal)
        
        # Rank signals by combined score
        for signal in filtered_signals:
            score = self._calculate_signal_score(signal, data[signal.symbol])
            signal.metadata['combined_score'] = score
        
        # Sort by score (descending)
        filtered_signals.sort(key=lambda x: x.metadata['combined_score'], reverse=True)
        
        # Limit number of signals
        max_signals = 10  # Maximum 10 positions at once
        return filtered_signals[:max_signals]
    
    def _calculate_signal_score(self, signal, symbol_data: pd.DataFrame) -> float:
        """
        Calculate a combined score for signal ranking
        """
        base_score = signal.strength
        
        # Adjust for consensus (more strategies agreeing = higher score)
        consensus_bonus = signal.metadata.get('consensus_strength', 1) * 0.1
        
        # Adjust for volatility (moderate volatility preferred)
        volatility = signal.metadata.get('volatility', 0.2)
        vol_adjustment = 1.0 - abs(volatility - 0.25)  # Optimal around 25%
        
        # Adjust for volume (higher volume = more liquid)
        volume = signal.metadata.get('avg_volume', 0)
        volume_adjustment = min(np.log(volume / 100000) / 5, 0.2) if volume > 0 else 0
        
        # Adjust for recent performance (momentum factor)
        recent_return = symbol_data['Close'].pct_change(5).iloc[-1]
        momentum_adjustment = 0.1 if (signal.signal_type.value > 0 and recent_return > 0) or \
                                    (signal.signal_type.value < 0 and recent_return < 0) else -0.05
        
        final_score = base_score + consensus_bonus + vol_adjustment + volume_adjustment + momentum_adjustment
        return max(0, min(1, final_score))  # Clamp between 0 and 1
    
    def update_strategy_performance(self, executed_trades: List):
        """
        Update strategy performance metrics for adaptive weighting
        """
        # This would analyze the performance of recent trades by strategy
        # and adjust strategy weights accordingly
        
        for trade in executed_trades:
            strategy_name = trade.get('strategy', 'unknown')
            if strategy_name in self.strategy_performance:
                # Update performance metrics
                # (Implementation would track returns, win rates, etc.)
                pass
        
        # Rebalance strategy weights based on recent performance
        self._rebalance_strategy_weights()
    
    def _rebalance_strategy_weights(self):
        """
        Rebalance strategy weights based on recent performance
        """
        # Calculate performance metrics for each strategy
        performance_scores = {}
        
        for strategy_name, perf in self.strategy_performance.items():
            # Simple scoring based on Sharpe ratio and win rate
            sharpe = perf.get('sharpe_ratio', 0)
            win_rate = perf.get('signals_profitable', 0) / max(perf.get('signals_generated', 1), 1)
            
            # Combined score
            score = (sharpe * 0.7) + (win_rate * 0.3)
            performance_scores[strategy_name] = max(score, 0.1)  # Minimum weight
        
        # Normalize weights
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] = performance_scores[strategy_name] / total_score
        
        logging.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def get_strategy_summary(self) -> Dict:
        """
        Get summary of strategy performance and weights
        """
        return {
            'weights': self.strategy_weights.copy(),
            'performance': self.strategy_performance.copy(),
            'total_signals_today': len(self.signals_history),
            'last_update': datetime.now()
        }

# Global strategy manager instance
strategy_manager = StrategyManager()
