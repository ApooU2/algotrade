"""
Unit tests for risk management components.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_management.risk_calculator import RiskCalculator
from risk_management.position_sizer import PositionSizer
from risk_management.risk_monitor import RiskMonitor
from config.config import RISK_CONFIG


class TestRiskCalculator(unittest.TestCase):
    """Test cases for Risk Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_calculator = RiskCalculator(RISK_CONFIG)
        
        # Create sample returns data
        np.random.seed(42)
        self.returns = pd.Series(
            np.random.normal(0.001, 0.02, 252),  # Daily returns
            index=pd.date_range(start='2023-01-01', periods=252, freq='D')
        )
        
        # Create sample portfolio data
        self.portfolio_data = {
            'AAPL': {'quantity': 100, 'price': 150.0, 'weight': 0.4},
            'GOOGL': {'quantity': 50, 'price': 120.0, 'weight': 0.3},
            'MSFT': {'quantity': 75, 'price': 80.0, 'weight': 0.3}
        }
    
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        var_95 = self.risk_calculator.calculate_var(self.returns, confidence=0.95)
        var_99 = self.risk_calculator.calculate_var(self.returns, confidence=0.99)
        
        self.assertIsInstance(var_95, float)
        self.assertIsInstance(var_99, float)
        self.assertLess(var_95, 0)  # VaR should be negative (loss)
        self.assertLess(var_99, var_95)  # 99% VaR should be more extreme
    
    def test_cvar_calculation(self):
        """Test Conditional Value at Risk calculation."""
        cvar_95 = self.risk_calculator.calculate_cvar(self.returns, confidence=0.95)
        cvar_99 = self.risk_calculator.calculate_cvar(self.returns, confidence=0.99)
        
        self.assertIsInstance(cvar_95, float)
        self.assertIsInstance(cvar_99, float)
        self.assertLess(cvar_95, 0)  # CVaR should be negative
        self.assertLess(cvar_99, cvar_95)  # 99% CVaR should be more extreme
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create price series with known drawdown
        prices = pd.Series([100, 110, 105, 95, 90, 100, 105])
        max_dd, dd_duration = self.risk_calculator.calculate_max_drawdown(prices)
        
        self.assertIsInstance(max_dd, float)
        self.assertIsInstance(dd_duration, int)
        self.assertLess(max_dd, 0)  # Max drawdown should be negative
        self.assertGreater(dd_duration, 0)  # Duration should be positive
    
    def test_correlation_risk(self):
        """Test correlation risk assessment."""
        # Create correlated returns
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100)
        })
        
        correlation_risk = self.risk_calculator.assess_correlation_risk(
            returns_data, self.portfolio_data
        )
        
        self.assertIsInstance(correlation_risk, dict)
        self.assertIn('avg_correlation', correlation_risk)
        self.assertIn('max_correlation', correlation_risk)
        self.assertIn('diversification_ratio', correlation_risk)
    
    def test_stress_testing(self):
        """Test stress testing scenarios."""
        stress_results = self.risk_calculator.run_stress_tests(
            self.portfolio_data, self.returns
        )
        
        self.assertIsInstance(stress_results, dict)
        
        expected_scenarios = ['market_crash', 'interest_rate_shock', 'volatility_spike']
        for scenario in expected_scenarios:
            self.assertIn(scenario, stress_results)
            self.assertIsInstance(stress_results[scenario], dict)


class TestPositionSizer(unittest.TestCase):
    """Test cases for Position Sizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.position_sizer = PositionSizer(RISK_CONFIG)
        
        # Mock portfolio
        self.mock_portfolio = Mock()
        self.mock_portfolio.total_value = 100000
        self.mock_portfolio.cash = 50000
        
        # Sample market data
        self.market_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98],
            'volume': [1000000, 1100000, 900000, 1200000, 800000]
        })
    
    def test_kelly_criterion_sizing(self):
        """Test Kelly Criterion position sizing."""
        win_rate = 0.6
        avg_win = 0.05
        avg_loss = -0.03
        
        position_size = self.position_sizer.kelly_criterion(
            portfolio_value=100000,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_price=150.0
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)
    
    def test_volatility_based_sizing(self):
        """Test volatility-based position sizing."""
        volatility = 0.25  # 25% annual volatility
        
        position_size = self.position_sizer.volatility_based(
            portfolio_value=100000,
            target_volatility=0.02,  # 2% target risk
            asset_volatility=volatility,
            current_price=150.0
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)
    
    def test_atr_based_sizing(self):
        """Test ATR-based position sizing."""
        # Calculate ATR manually for test data
        high = pd.Series([101, 102, 100, 103, 99])
        low = pd.Series([99, 100, 98, 101, 97])
        close = pd.Series([100, 101, 99, 102, 98])
        
        position_size = self.position_sizer.atr_based(
            portfolio_value=100000,
            risk_per_trade=0.01,  # 1% risk per trade
            atr_period=14,
            high=high,
            low=low,
            close=close,
            current_price=100.0
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)
    
    def test_dynamic_sizing(self):
        """Test dynamic position sizing."""
        recent_performance = pd.Series([0.02, -0.01, 0.03, -0.005, 0.015])
        
        position_size = self.position_sizer.dynamic_sizing(
            base_position_size=1000,
            recent_performance=recent_performance,
            market_regime='trending',
            volatility=0.20
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)


class TestRiskMonitor(unittest.TestCase):
    """Test cases for Risk Monitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RISK_CONFIG.copy()
        self.risk_monitor = RiskMonitor(self.config)
        
        # Mock notification manager
        self.mock_notifier = Mock()
        self.risk_monitor.notification_manager = self.mock_notifier
        
        # Sample portfolio
        self.portfolio = {
            'total_value': 100000,
            'cash': 20000,
            'positions': {
                'AAPL': {'quantity': 100, 'current_price': 150.0, 'unrealized_pnl': 1000},
                'GOOGL': {'quantity': 50, 'current_price': 120.0, 'unrealized_pnl': -500}
            }
        }
    
    def test_portfolio_risk_assessment(self):
        """Test portfolio risk assessment."""
        risk_metrics = self.risk_monitor.assess_portfolio_risk(self.portfolio)
        
        self.assertIsInstance(risk_metrics, dict)
        
        expected_metrics = [
            'total_exposure', 'concentration_risk', 'leverage',
            'unrealized_pnl', 'risk_score'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, risk_metrics)
    
    def test_position_limits_check(self):
        """Test position limits checking."""
        violations = self.risk_monitor.check_position_limits(self.portfolio)
        
        self.assertIsInstance(violations, list)
        # Each violation should be a dictionary with details
        for violation in violations:
            self.assertIsInstance(violation, dict)
            self.assertIn('type', violation)
            self.assertIn('message', violation)
    
    def test_drawdown_monitoring(self):
        """Test drawdown monitoring."""
        # Create equity curve with drawdown
        equity_curve = pd.Series([
            100000, 105000, 103000, 95000, 90000, 95000, 98000
        ])
        
        alert_triggered = self.risk_monitor.monitor_drawdown(equity_curve)
        
        self.assertIsInstance(alert_triggered, bool)
    
    def test_real_time_monitoring(self):
        """Test real-time risk monitoring."""
        market_data = {
            'AAPL': {'price': 145.0, 'volume': 1000000, 'volatility': 0.25},
            'GOOGL': {'price': 125.0, 'volume': 800000, 'volatility': 0.30}
        }
        
        alerts = self.risk_monitor.real_time_monitoring(self.portfolio, market_data)
        
        self.assertIsInstance(alerts, list)
    
    def test_compliance_checking(self):
        """Test compliance rule checking."""
        compliance_results = self.risk_monitor.check_compliance(self.portfolio)
        
        self.assertIsInstance(compliance_results, dict)
        self.assertIn('compliant', compliance_results)
        self.assertIn('violations', compliance_results)
        self.assertIsInstance(compliance_results['compliant'], bool)
        self.assertIsInstance(compliance_results['violations'], list)


if __name__ == '__main__':
    unittest.main()
