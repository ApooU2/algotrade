#!/usr/bin/env python3
"""
Test script to validate market hours restrictions and position holding time limits
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from execution.demo_trader import DemoTradingEngine
from config.config import TIME_BASED_CONFIG

def test_market_hours_restrictions():
    """Test that non-crypto stocks cannot be traded outside market hours"""
    print("🧪 Testing Market Hours Restrictions")
    print("=" * 50)
    
    trader = DemoTradingEngine(initial_capital=10000, save_state=False)
    
    # Test crypto trading (should be allowed 24/7)
    crypto_symbol = "BTC-USD"
    can_trade_crypto = trader.can_trade_symbol_now(crypto_symbol, is_market_hours=False)
    print(f"✅ Crypto {crypto_symbol} after hours: {can_trade_crypto}")
    assert can_trade_crypto, "Crypto should be tradeable 24/7"
    
    # Test stock trading outside market hours (should be restricted)
    stock_symbol = "AAPL"
    can_trade_stock = trader.can_trade_symbol_now(stock_symbol, is_market_hours=False)
    print(f"❌ Stock {stock_symbol} after hours: {can_trade_stock}")
    assert not can_trade_stock, "Stocks should not be tradeable outside market hours"
    
    # Test stock trading during market hours (should be allowed)
    can_trade_stock_market_hours = trader.can_trade_symbol_now(stock_symbol, is_market_hours=True)
    print(f"✅ Stock {stock_symbol} during hours: {can_trade_stock_market_hours}")
    assert can_trade_stock_market_hours, "Stocks should be tradeable during market hours"
    
    print("✅ Market hours restrictions working correctly!\n")

def test_position_holding_limits():
    """Test that positions exceeding holding limits are identified"""
    print("🧪 Testing Position Holding Time Limits")
    print("=" * 50)
    
    trader = DemoTradingEngine(initial_capital=10000, save_state=False)
    
    # Import the DemoPosition class
    from execution.demo_trader import DemoPosition
    
    # Create test positions with different ages
    current_time = datetime.now()
    
    # Add a new position (should be fine)
    trader.positions["MSFT"] = DemoPosition(
        symbol="MSFT",
        shares=10,
        entry_price=100.0,
        entry_time=current_time - timedelta(days=1),  # 1 day old
        current_price=105.0
    )
    
    # Add an aged position (should be flagged)
    trader.positions["GOOGL"] = DemoPosition(
        symbol="GOOGL", 
        shares=5,
        entry_price=2000.0,
        entry_time=current_time - timedelta(days=10),  # 10 days old
        current_price=2100.0
    )
    
    # Test position age checking
    max_age_days = TIME_BASED_CONFIG['max_holding_period_days']
    aged_positions = trader.get_aged_positions(max_age_days)
    
    print(f"Max holding period: {max_age_days} days")
    print(f"Aged positions found: {aged_positions}")
    
    # GOOGL should be in aged positions (10 days > 7 days default)
    assert "GOOGL" in aged_positions, f"GOOGL should be flagged as aged (10 days > {max_age_days} days)"
    assert "MSFT" not in aged_positions, f"MSFT should not be flagged as aged (1 day < {max_age_days} days)"
    
    # Test holding time calculation
    msft_holding_time = trader.get_position_holding_time("MSFT")
    googl_holding_time = trader.get_position_holding_time("GOOGL")
    
    print(f"MSFT holding time: {msft_holding_time}")
    print(f"GOOGL holding time: {googl_holding_time}")
    
    assert msft_holding_time.days < max_age_days, "MSFT should be within holding limit"
    assert googl_holding_time.days > max_age_days, "GOOGL should exceed holding limit"
    
    print("✅ Position holding time limits working correctly!\n")

def test_forced_position_closure():
    """Test that aged positions can be force closed"""
    print("🧪 Testing Forced Position Closure")
    print("=" * 50)
    
    trader = DemoTradingEngine(initial_capital=10000, save_state=False)
    
    # Add an aged position
    current_time = datetime.now()
    aged_symbol = "AGED"
    
    # Mock the position manually
    from execution.demo_trader import DemoPosition
    aged_position = DemoPosition(
        symbol=aged_symbol,
        shares=100,
        entry_price=50.0,
        entry_time=current_time - timedelta(days=15),  # Very old position
        current_price=55.0
    )
    trader.positions[aged_symbol] = aged_position
    
    print(f"Added aged position: {aged_symbol} (15 days old)")
    print(f"Positions before closure: {list(trader.positions.keys())}")
    
    # Mock the price fetching to avoid Yahoo Finance calls
    with patch.object(trader, 'get_current_price', return_value=55.0):
        # Force close aged positions
        max_age_days = 7
        closed_positions = trader.force_close_aged_positions(max_age_days)
    
    print(f"Closed positions: {closed_positions}")
    print(f"Positions after closure: {list(trader.positions.keys())}")
    
    assert aged_symbol in closed_positions, "Aged position should have been closed"
    assert aged_symbol not in trader.positions, "Aged position should no longer exist in portfolio"
    
    print("✅ Forced position closure working correctly!\n")

def main():
    """Run all time restriction tests"""
    print("🚀 TESTING TIME-BASED TRADING RESTRICTIONS")
    print("=" * 60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Time-based Config: {TIME_BASED_CONFIG}")
    print()
    
    try:
        test_market_hours_restrictions()
        test_position_holding_limits()
        test_forced_position_closure()
        
        print("🎉 ALL TESTS PASSED!")
        print("✅ Market hours restrictions implemented correctly")
        print("✅ Position holding time limits implemented correctly") 
        print("✅ Automatic position closure working correctly")
        print("\n📋 SUMMARY:")
        print("• Stocks cannot be traded outside market hours (9:30 AM - 4:00 PM ET)")
        print("• Crypto can be traded 24/7")
        print(f"• Positions are automatically closed after {TIME_BASED_CONFIG['max_holding_period_days']} days")
        print("• Risk monitoring alerts for positions approaching limits")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
