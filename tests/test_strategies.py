"""
Unit tests for trading strategies.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy
from config.config import MEAN_REVERSION_CONFIG, MOMENTUM_CONFIG, BREAKOUT_CONFIG


class TestMeanReversionStrategy(unittest.TestCase):
    """Test cases for Mean Reversion Strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MeanReversionStrategy(MEAN_REVERSION_CONFIG)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.name, "Mean Reversion")
        self.assertIsInstance(self.strategy.config, dict)
    
    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.sample_data)
        
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.sample_data))
        self.assertTrue(all(signal in [-1, 0, 1] for signal in signals))
    
    def test_calculate_indicators(self):
        """Test technical indicator calculation."""
        indicators = self.strategy.calculate_indicators(self.sample_data)
        
        required_indicators = ['bb_upper', 'bb_lower', 'bb_middle', 'rsi']
        for indicator in required_indicators:
            self.assertIn(indicator, indicators.columns)
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        mock_portfolio = Mock()
        mock_portfolio.total_value = 100000
        mock_portfolio.get_position.return_value = None
        
        position_size = self.strategy.calculate_position_size(
            symbol='AAPL',
            signal=1,
            current_price=150.0,
            portfolio=mock_portfolio
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)


class TestMomentumStrategy(unittest.TestCase):
    """Test cases for Momentum Strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MomentumStrategy(MOMENTUM_CONFIG)
        
        # Create trending sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        trend = np.linspace(0, 50, len(dates))
        noise = np.random.randn(len(dates)) * 2
        prices = 100 + trend + noise
        
        self.sample_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.name, "Momentum")
    
    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.sample_data)
        
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.sample_data))
        self.assertTrue(all(signal in [-1, 0, 1] for signal in signals))
    
    def test_macd_calculation(self):
        """Test MACD indicator calculation."""
        indicators = self.strategy.calculate_indicators(self.sample_data)
        
        required_indicators = ['macd', 'macd_signal', 'macd_histogram', 'trend_ma']
        for indicator in required_indicators:
            self.assertIn(indicator, indicators.columns)


class TestBreakoutStrategy(unittest.TestCase):
    """Test cases for Breakout Strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = BreakoutStrategy(BREAKOUT_CONFIG)
        
        # Create consolidating sample data with breakouts
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create consolidation periods followed by breakouts
        prices = []
        base_price = 100
        
        for i in range(len(dates)):
            if i < len(dates) // 3:
                # Consolidation period
                prices.append(base_price + np.random.randn() * 2)
            elif i < 2 * len(dates) // 3:
                # Breakout period
                base_price += 0.5
                prices.append(base_price + np.random.randn() * 3)
            else:
                # Another consolidation
                prices.append(base_price + np.random.randn() * 2)
        
        self.sample_data = pd.DataFrame({
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.name, "Breakout")
    
    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.sample_data)
        
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.sample_data))
        self.assertTrue(all(signal in [-1, 0, 1] for signal in signals))
    
    def test_consolidation_detection(self):
        """Test consolidation pattern detection."""
        indicators = self.strategy.calculate_indicators(self.sample_data)
        
        self.assertIn('consolidation', indicators.columns)
        self.assertIn('breakout_level', indicators.columns)


class TestStrategyPerformance(unittest.TestCase):
    """Test strategy performance metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategies = [
            MeanReversionStrategy(MEAN_REVERSION_CONFIG),
            MomentumStrategy(MOMENTUM_CONFIG),
            BreakoutStrategy(BREAKOUT_CONFIG)
        ]
        
        # Create sample trade history
        self.trades = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'symbol': ['AAPL'] * 100,
            'side': np.random.choice(['buy', 'sell'], 100),
            'quantity': np.random.randint(10, 100, 100),
            'price': 150 + np.random.randn(100) * 10,
            'commission': 1.0
        })
    
    def test_performance_calculation(self):
        """Test performance metrics calculation."""
        for strategy in self.strategies:
            metrics = strategy.calculate_performance_metrics(self.trades)
            
            required_metrics = [
                'total_return', 'sharpe_ratio', 'max_drawdown',
                'win_rate', 'total_trades', 'avg_trade_return'
            ]
            
            for metric in required_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        for strategy in self.strategies:
            returns = np.random.randn(252) * 0.02  # Daily returns
            
            metrics = strategy.calculate_risk_metrics(returns)
            
            self.assertIn('volatility', metrics)
            self.assertIn('var_95', metrics)
            self.assertIn('var_99', metrics)
            self.assertGreater(metrics['volatility'], 0)


if __name__ == '__main__':
    unittest.main()
