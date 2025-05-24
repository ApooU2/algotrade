#!/usr/bin/env python3
"""
Validation script to verify the trading bot setup and functionality.
"""

import os
import sys
import traceback
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    required_modules = {
        'Core Data Science': ['pandas', 'numpy', 'scipy'],
        'Trading APIs': ['alpaca_trade_api', 'yfinance'],
        'Technical Analysis': ['talib'],
        'Machine Learning': ['sklearn', 'ydf'],
        'Visualization': ['plotly', 'matplotlib', 'seaborn'],
        'Utilities': ['requests', 'sqlalchemy', 'schedule', 'jinja2']
    }
    
    all_passed = True
    
    for category, modules in required_modules.items():
        print(f"\n  📦 {category}:")
        for module in modules:
            try:
                __import__(module)
                print(f"    ✅ {module}")
            except ImportError as e:
                print(f"    ❌ {module} - {e}")
                all_passed = False
    
    return all_passed


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from config.config import (
            TRADING_CONFIG, RISK_CONFIG, BACKTESTING_CONFIG,
            MEAN_REVERSION_CONFIG, MOMENTUM_CONFIG, ML_CONFIG
        )
        
        configs = {
            'TRADING_CONFIG': TRADING_CONFIG,
            'RISK_CONFIG': RISK_CONFIG,
            'BACKTESTING_CONFIG': BACKTESTING_CONFIG,
            'MEAN_REVERSION_CONFIG': MEAN_REVERSION_CONFIG,
            'MOMENTUM_CONFIG': MOMENTUM_CONFIG,
            'ML_CONFIG': ML_CONFIG
        }
        
        for name, config in configs.items():
            if config:
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name} - empty or invalid")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def test_data_access():
    """Test data access functionality."""
    print("\n📊 Testing data access...")
    
    try:
        from data.data_manager import DataManager
        import pandas as pd
        
        # Initialize data manager
        data_manager = DataManager()
        print("  ✅ DataManager initialized")
        
        # Test historical data fetch
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = data_manager.get_historical_data(
                ['AAPL'], 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if isinstance(data, dict) and 'AAPL' in data and not data['AAPL'].empty:
                print(f"  ✅ Historical data fetch (AAPL, {len(data['AAPL'])} records)")
                aapl_data = data['AAPL']
            else:
                print("  ❌ Historical data fetch - empty result")
                return False
                
        except Exception as e:
            print(f"  ❌ Historical data fetch failed: {e}")
            return False
        
        # Test technical indicators
        try:
            indicators = data_manager.calculate_technical_indicators(aapl_data)
            if isinstance(indicators, pd.DataFrame) and not indicators.empty:
                print(f"  ✅ Technical indicators ({len(indicators.columns)} indicators)")
            else:
                print("  ❌ Technical indicators calculation failed")
                return False
                
        except Exception as e:
            print(f"  ❌ Technical indicators error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data access error: {e}")
        print(f"  📋 Full error: {traceback.format_exc()}")
        return False


def test_strategies():
    """Test strategy initialization and basic functionality."""
    print("\n🎯 Testing strategies...")
    
    strategies_to_test = [
        ('Mean Reversion', 'strategies.mean_reversion', 'MeanReversionStrategy'),
        ('Momentum', 'strategies.momentum', 'MomentumStrategy'),
        ('Breakout', 'strategies.breakout', 'BreakoutStrategy'),
        ('ML Ensemble', 'strategies.ml_ensemble', 'MLEnsembleStrategy'),
    ]
    
    all_passed = True
    
    for name, module_name, class_name in strategies_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            
            # Try to initialize strategy
            if name == 'Mean Reversion':
                from config.config import MEAN_REVERSION_CONFIG
                strategy = strategy_class(MEAN_REVERSION_CONFIG)
            elif name == 'Momentum':
                from config.config import MOMENTUM_CONFIG
                strategy = strategy_class(MOMENTUM_CONFIG)
            elif name == 'Breakout':
                from config.config import BREAKOUT_CONFIG
                strategy = strategy_class(BREAKOUT_CONFIG)
            elif name == 'ML Ensemble':
                from config.config import ML_CONFIG
                strategy = strategy_class(ML_CONFIG)
            
            print(f"  ✅ {name} strategy")
            
        except Exception as e:
            print(f"  ❌ {name} strategy - {e}")
            all_passed = False
    
    return all_passed


def test_risk_management():
    """Test risk management components."""
    print("\n🛡️  Testing risk management...")
    
    try:
        from risk_management.risk_calculator import RiskCalculator
        from risk_management.position_sizer import PositionSizer
        from risk_management.risk_monitor import RiskMonitor
        from config.config import RISK_CONFIG
        
        # Test RiskCalculator
        risk_calc = RiskCalculator(RISK_CONFIG)
        print("  ✅ RiskCalculator")
        
        # Test PositionSizer
        position_sizer = PositionSizer(RISK_CONFIG)
        print("  ✅ PositionSizer")
        
        # Test RiskMonitor
        risk_monitor = RiskMonitor(RISK_CONFIG)
        print("  ✅ RiskMonitor")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Risk management error: {e}")
        return False


def test_backtesting():
    """Test backtesting engine."""
    print("\n📈 Testing backtesting engine...")
    
    try:
        from backtesting.backtest_engine import BacktestEngine
        from config.config import BACKTESTING_CONFIG
        
        # Initialize backtest engine
        engine = BacktestEngine(BACKTESTING_CONFIG)
        print("  ✅ BacktestEngine initialized")
        
        # Test basic functionality (without running full backtest)
        if hasattr(engine, 'portfolio') and hasattr(engine, 'initial_capital'):
            print("  ✅ BacktestEngine basic functionality")
        else:
            print("  ❌ BacktestEngine missing required attributes")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backtesting error: {e}")
        return False


def test_visualization():
    """Test visualization components."""
    print("\n📊 Testing visualization...")
    
    try:
        from visualization.performance_visualizer import PerformanceVisualizer
        from visualization.trading_charts import TradingCharts
        from visualization.risk_visualizer import RiskVisualizer
        
        # Test initialization
        perf_viz = PerformanceVisualizer()
        print("  ✅ PerformanceVisualizer")
        
        trading_charts = TradingCharts()
        print("  ✅ TradingCharts")
        
        risk_viz = RiskVisualizer()
        print("  ✅ RiskVisualizer")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization error: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\n🔧 Testing utilities...")
    
    try:
        from utils.helpers import calculate_returns, calculate_sharpe_ratio
        from utils.decorators import timing_decorator, retry_on_failure
        from utils.notifications import EmailNotifier, NotificationManager
        
        print("  ✅ Helper functions")
        print("  ✅ Decorators")
        print("  ✅ Notifications")
        
        # Test basic utility function
        import pandas as pd
        import numpy as np
        
        prices = pd.Series([100, 102, 101, 105, 103])
        returns = calculate_returns(prices)
        
        if isinstance(returns, pd.Series) and len(returns) == len(prices) - 1:
            print("  ✅ Returns calculation")
        else:
            print("  ❌ Returns calculation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utilities error: {e}")
        return False


def test_environment_variables():
    """Test environment variable loading."""
    print("\n🌍 Testing environment variables...")
    
    env_file = '.env'
    if os.path.exists(env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("  ✅ .env file loaded")
            
            # Check for critical variables
            critical_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
            missing_vars = []
            
            for var in critical_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"  ⚠️  Missing environment variables: {missing_vars}")
                print("     (This is expected if you haven't configured them yet)")
            else:
                print("  ✅ Critical environment variables found")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Environment loading error: {e}")
            return False
    else:
        print("  ⚠️  .env file not found (run setup.py first)")
        return False


def run_validation():
    """Run all validation tests."""
    print("🔍 Algorithmic Trading Bot Validation")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Access", test_data_access),
        ("Strategies", test_strategies),
        ("Risk Management", test_risk_management),
        ("Backtesting", test_backtesting),
        ("Visualization", test_visualization),
        ("Utilities", test_utilities),
        ("Environment Variables", test_environment_variables),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your trading bot is ready to use.")
        print("\n📋 Next steps:")
        print("  1. Configure your .env file with API keys")
        print("  2. Run backtesting: python main.py --backtest-only")
        print("  3. Start paper trading: python main.py --paper-trading")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues before proceeding.")
        print("\n💡 Common fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Install TA-Lib (see README.md for instructions)")
        print("  - Check your Python version (3.8+ required)")
    
    return passed == total


if __name__ == '__main__':
    success = run_validation()
    sys.exit(0 if success else 1)
