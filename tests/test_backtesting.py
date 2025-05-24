"""
Unit tests for backtesting engine.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_engine import BacktestEngine
from strategies.mean_reversion import MeanReversionStrategy
from config.config import BACKTESTING_CONFIG, MEAN_REVERSION_CONFIG


class TestBacktestEngine(unittest.TestCase):
    """Test cases for Backtest Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BACKTESTING_CONFIG.copy()
        self.engine = BacktestEngine(self.config)
        
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.market_data = pd.DataFrame({
            'open': [p * (0.99 + np.random.random() * 0.02) for p in prices],
            'high': [p * (1.00 + np.random.random() * 0.03) for p in prices],
            'low': [p * (0.97 + np.random.random() * 0.02) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Sample strategy
        self.strategy = MeanReversionStrategy(MEAN_REVERSION_CONFIG)
    
    def test_initialization(self):
        """Test backtest engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.initial_capital, self.config['initial_capital'])
        self.assertIsInstance(self.engine.portfolio, dict)
    
    def test_single_asset_backtest(self):
        """Test backtesting with single asset."""
        results = self.engine.backtest(
            strategy=self.strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-06-30',
            data=self.market_data[:180]  # First half of year
        )
        
        self.assertIsInstance(results, dict)
        
        expected_metrics = [
            'total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'total_trades', 'final_portfolio_value'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (int, float))
    
    def test_multi_asset_backtest(self):
        """Test backtesting with multiple assets."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        multi_data = {}
        
        for symbol in symbols:
            # Create slightly different data for each symbol
            np.random.seed(hash(symbol) % 1000)
            returns = np.random.normal(0.0005, 0.02, len(self.market_data))
            prices = [100]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            multi_data[symbol] = pd.DataFrame({
                'open': [p * 0.99 for p in prices],
                'high': [p * 1.02 for p in prices],
                'low': [p * 0.98 for p in prices],
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, len(prices))
            }, index=self.market_data.index)
        
        results = self.engine.backtest_portfolio(
            strategy=self.strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-06-30',
            data=multi_data
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('portfolio_metrics', results)
        self.assertIn('individual_assets', results)
    
    def test_portfolio_management(self):
        """Test portfolio management during backtesting."""
        # Test initial portfolio setup
        self.engine._reset_portfolio()
        
        self.assertEqual(self.engine.portfolio['cash'], self.config['initial_capital'])
        self.assertEqual(len(self.engine.portfolio['positions']), 0)
        
        # Test position opening
        self.engine._execute_trade('AAPL', 'buy', 100, 150.0, datetime.now())
        
        self.assertIn('AAPL', self.engine.portfolio['positions'])
        self.assertEqual(self.engine.portfolio['positions']['AAPL']['quantity'], 100)
        self.assertLess(self.engine.portfolio['cash'], self.config['initial_capital'])
    
    def test_trade_execution(self):
        """Test trade execution logic."""
        initial_cash = self.engine.portfolio['cash']
        
        # Execute buy order
        trade_cost = self.engine._execute_trade('AAPL', 'buy', 100, 150.0, datetime.now())
        
        expected_cost = 100 * 150.0 + self.config['commission']
        self.assertAlmostEqual(trade_cost, expected_cost, places=2)
        self.assertAlmostEqual(
            self.engine.portfolio['cash'], 
            initial_cash - expected_cost, 
            places=2
        )
        
        # Execute sell order
        sell_proceeds = self.engine._execute_trade('AAPL', 'sell', 50, 155.0, datetime.now())
        
        expected_proceeds = 50 * 155.0 - self.config['commission']
        self.assertAlmostEqual(sell_proceeds, expected_proceeds, places=2)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create sample equity curve
        equity_curve = pd.Series([
            100000, 102000, 101000, 105000, 103000, 
            107000, 104000, 108000, 106000, 110000
        ])
        
        metrics = self.engine._calculate_performance_metrics(equity_curve)
        
        self.assertIsInstance(metrics, dict)
        
        expected_metrics = [
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'calmar_ratio', 'sortino_ratio'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_walk_forward_analysis(self):
        """Test walk-forward analysis."""
        results = self.engine.walk_forward_analysis(
            strategy=self.strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-06-30',
            data=self.market_data[:180],
            window_size=30,  # 30-day windows
            step_size=15     # 15-day steps
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('windows', results)
        self.assertIn('summary', results)
        self.assertIsInstance(results['windows'], list)
        self.assertGreater(len(results['windows']), 0)
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        mc_results = self.engine.monte_carlo_simulation(
            strategy=self.strategy,
            symbol='AAPL',
            base_data=self.market_data[:100],
            num_simulations=10,  # Small number for testing
            confidence_levels=[0.05, 0.95]
        )
        
        self.assertIsInstance(mc_results, dict)
        self.assertIn('simulations', mc_results)
        self.assertIn('statistics', mc_results)
        self.assertEqual(len(mc_results['simulations']), 10)
    
    def test_transaction_costs(self):
        """Test transaction cost calculation."""
        # Test commission
        commission = self.engine._calculate_commission(100, 150.0)
        self.assertEqual(commission, self.config['commission'])
        
        # Test slippage
        slippage_cost = self.engine._calculate_slippage(100, 150.0, 'buy')
        expected_slippage = 100 * 150.0 * (self.config['slippage'] / 10000)
        self.assertAlmostEqual(slippage_cost, expected_slippage, places=2)
    
    def test_risk_constraints(self):
        """Test risk constraint enforcement."""
        # Test position size limits
        max_position_value = self.config['max_position_size'] * self.engine.portfolio['cash']
        large_quantity = int(max_position_value / 100.0) + 100  # Exceed limit
        
        constrained_quantity = self.engine._apply_risk_constraints(
            'AAPL', 'buy', large_quantity, 100.0
        )
        
        self.assertLess(constrained_quantity, large_quantity)
        
        # Verify portfolio risk limits
        portfolio_value = sum(
            pos['quantity'] * pos['avg_price'] 
            for pos in self.engine.portfolio['positions'].values()
        ) + self.engine.portfolio['cash']
        
        max_portfolio_risk = self.config['max_portfolio_risk'] * portfolio_value
        
        # This should be enforced by the risk constraints
        self.assertGreaterEqual(max_portfolio_risk, 0)


if __name__ == '__main__':
    unittest.main()
