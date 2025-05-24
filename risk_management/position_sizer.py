"""
Position sizing module for optimal position size calculation based on risk management principles.
Implements various position sizing methods including Kelly Criterion, fixed fractional, and volatility-based sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


class PositionSizer:
    """
    Advanced position sizing system with multiple methodologies and risk controls.
    """
    
    def __init__(self, 
                 portfolio_value: float,
                 max_position_size: float = 0.1,
                 max_portfolio_risk: float = 0.02):
        """
        Initialize the position sizer.
        
        Args:
            portfolio_value: Current portfolio value
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_risk: Maximum portfolio risk per trade
        """
        self.portfolio_value = portfolio_value
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
    
    def kelly_criterion_size(self, 
                           win_rate: float,
                           avg_win: float,
                           avg_loss: float,
                           kelly_fraction: float = 0.25) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            kelly_fraction: Fraction of Kelly to use (for safety)
            
        Returns:
            Position size as fraction of portfolio
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_f = (b * p - q) / b
        
        # Apply safety fraction and constraints
        position_size = max(0, kelly_f * kelly_fraction)
        position_size = min(position_size, self.max_position_size)
        
        return position_size
    
    def volatility_based_size(self, 
                            price: float,
                            volatility: float,
                            target_volatility: float = 0.2) -> int:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            price: Current asset price
            volatility: Asset's annual volatility
            target_volatility: Target portfolio volatility contribution
            
        Returns:
            Number of shares to trade
        """
        if volatility <= 0 or price <= 0:
            return 0
        
        # Calculate position value for target volatility
        target_position_value = (self.portfolio_value * target_volatility) / volatility
        
        # Apply maximum position size constraint
        max_position_value = self.portfolio_value * self.max_position_size
        position_value = min(target_position_value, max_position_value)
        
        # Convert to shares
        shares = int(position_value / price)
        
        return shares
    
    def fixed_fractional_size(self, 
                            price: float,
                            risk_fraction: float = 0.01) -> int:
        """
        Calculate position size using fixed fractional method.
        
        Args:
            price: Current asset price
            risk_fraction: Fraction of portfolio to risk
            
        Returns:
            Number of shares to trade
        """
        if price <= 0:
            return 0
        
        risk_amount = self.portfolio_value * min(risk_fraction, self.max_portfolio_risk)
        position_value = min(risk_amount / 0.02, self.portfolio_value * self.max_position_size)  # Assume 2% stop loss
        
        shares = int(position_value / price)
        return shares
    
    def atr_based_size(self, 
                      price: float,
                      atr: float,
                      atr_multiplier: float = 2.0) -> int:
        """
        Calculate position size based on Average True Range (ATR).
        
        Args:
            price: Current asset price
            atr: Average True Range value
            atr_multiplier: ATR multiplier for stop loss
            
        Returns:
            Number of shares to trade
        """
        if price <= 0 or atr <= 0:
            return 0
        
        # Calculate stop loss distance
        stop_distance = atr * atr_multiplier
        
        # Risk amount per share
        risk_per_share = stop_distance
        
        # Total risk amount
        total_risk = self.portfolio_value * self.max_portfolio_risk
        
        # Calculate shares
        shares = int(total_risk / risk_per_share) if risk_per_share > 0 else 0
        
        # Apply position size limit
        max_shares = int((self.portfolio_value * self.max_position_size) / price)
        shares = min(shares, max_shares)
        
        return shares
    
    def optimal_f_size(self, 
                      trade_returns: List[float],
                      price: float) -> int:
        """
        Calculate position size using Optimal F method.
        
        Args:
            trade_returns: Historical trade returns
            price: Current asset price
            
        Returns:
            Number of shares to trade
        """
        if not trade_returns or price <= 0:
            return 0
        
        def terminal_wealth_ratio(f, returns):
            """Calculate terminal wealth ratio for given f."""
            if f <= 0:
                return 0
            
            twr = 1.0
            for ret in returns:
                new_value = 1 + f * ret
                if new_value <= 0:
                    return 0
                twr *= new_value
            
            return twr ** (1 / len(returns))
        
        # Find optimal f
        result = minimize_scalar(
            lambda f: -terminal_wealth_ratio(f, trade_returns),
            bounds=(0, 1),
            method='bounded'
        )
        
        optimal_f = result.x if result.success else 0
        
        # Apply safety constraints
        safe_f = min(optimal_f * 0.5, self.max_position_size)  # Use 50% of optimal f
        
        position_value = self.portfolio_value * safe_f
        shares = int(position_value / price)
        
        return shares
    
    def monte_carlo_size(self, 
                        price: float,
                        expected_return: float,
                        volatility: float,
                        num_simulations: int = 10000,
                        confidence_level: float = 0.95) -> int:
        """
        Calculate position size using Monte Carlo simulation.
        
        Args:
            price: Current asset price
            expected_return: Expected daily return
            volatility: Daily volatility
            num_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for risk assessment
            
        Returns:
            Number of shares to trade
        """
        if price <= 0 or volatility <= 0:
            return 0
        
        # Generate random returns
        returns = np.random.normal(expected_return, volatility, num_simulations)
        
        # Calculate VaR at confidence level
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Position size based on VaR
        if var < 0:  # Only proceed if VaR is negative (loss)
            risk_amount = self.portfolio_value * self.max_portfolio_risk
            position_value = risk_amount / abs(var)
            
            # Apply position size constraint
            max_position_value = self.portfolio_value * self.max_position_size
            position_value = min(position_value, max_position_value)
            
            shares = int(position_value / price)
            return shares
        
        return 0
    
    def correlation_adjusted_size(self, 
                                price: float,
                                base_size: int,
                                correlation_matrix: pd.DataFrame,
                                symbol: str,
                                current_positions: Dict[str, float]) -> int:
        """
        Adjust position size based on correlation with existing positions.
        
        Args:
            price: Current asset price
            base_size: Base position size before correlation adjustment
            correlation_matrix: Correlation matrix of assets
            symbol: Symbol to trade
            current_positions: Dictionary of current positions {symbol: weight}
            
        Returns:
            Correlation-adjusted position size
        """
        if symbol not in correlation_matrix.columns or not current_positions:
            return base_size
        
        # Calculate portfolio correlation
        portfolio_correlation = 0
        total_weight = sum(abs(weight) for weight in current_positions.values())
        
        if total_weight > 0:
            for existing_symbol, weight in current_positions.items():
                if existing_symbol in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[symbol, existing_symbol]
                    portfolio_correlation += (abs(weight) / total_weight) * abs(correlation)
        
        # Adjust size based on correlation
        correlation_factor = 1 - min(portfolio_correlation, 0.8)  # Max 80% reduction
        adjusted_size = int(base_size * correlation_factor)
        
        return adjusted_size
    
    def multi_strategy_size(self, 
                          strategies_data: Dict[str, Dict],
                          symbol: str,
                          price: float) -> Dict[str, int]:
        """
        Calculate position sizes for multiple strategies trading the same asset.
        
        Args:
            strategies_data: Dictionary with strategy data
            symbol: Symbol to trade
            price: Current asset price
            
        Returns:
            Dictionary of position sizes per strategy
        """
        position_sizes = {}
        total_exposure = 0
        
        for strategy_name, data in strategies_data.items():
            # Get strategy-specific parameters
            confidence = data.get('confidence', 0.5)
            volatility = data.get('volatility', 0.2)
            expected_return = data.get('expected_return', 0.01)
            
            # Calculate base size using volatility method
            base_size = self.volatility_based_size(price, volatility, 
                                                 target_volatility=0.1 * confidence)
            
            # Adjust for strategy confidence
            adjusted_size = int(base_size * confidence)
            
            position_sizes[strategy_name] = adjusted_size
            total_exposure += adjusted_size * price
        
        # Scale down if total exposure exceeds limits
        max_total_value = self.portfolio_value * self.max_position_size
        if total_exposure > max_total_value:
            scale_factor = max_total_value / total_exposure
            position_sizes = {strategy: int(size * scale_factor) 
                            for strategy, size in position_sizes.items()}
        
        return position_sizes
    
    def dynamic_size_adjustment(self, 
                              base_size: int,
                              market_conditions: Dict[str, float],
                              performance_metrics: Dict[str, float]) -> int:
        """
        Dynamically adjust position size based on market conditions and performance.
        
        Args:
            base_size: Base position size
            market_conditions: Dictionary with market condition indicators
            performance_metrics: Dictionary with recent performance metrics
            
        Returns:
            Adjusted position size
        """
        adjustment_factor = 1.0
        
        # Market volatility adjustment
        if 'market_volatility' in market_conditions:
            vol = market_conditions['market_volatility']
            if vol > 0.3:  # High volatility
                adjustment_factor *= 0.7
            elif vol < 0.1:  # Low volatility
                adjustment_factor *= 1.2
        
        # Trend strength adjustment
        if 'trend_strength' in market_conditions:
            trend = market_conditions['trend_strength']
            if abs(trend) > 0.7:  # Strong trend
                adjustment_factor *= 1.1
            elif abs(trend) < 0.3:  # Weak trend
                adjustment_factor *= 0.9
        
        # Recent performance adjustment
        if 'recent_sharpe' in performance_metrics:
            sharpe = performance_metrics['recent_sharpe']
            if sharpe > 1.5:  # Good performance
                adjustment_factor *= 1.1
            elif sharpe < 0.5:  # Poor performance
                adjustment_factor *= 0.8
        
        # Drawdown adjustment
        if 'current_drawdown' in performance_metrics:
            drawdown = abs(performance_metrics['current_drawdown'])
            if drawdown > 0.1:  # Large drawdown
                adjustment_factor *= 0.6
            elif drawdown > 0.05:  # Moderate drawdown
                adjustment_factor *= 0.8
        
        # Apply adjustment
        adjusted_size = int(base_size * adjustment_factor)
        
        # Ensure minimum and maximum bounds
        max_size = int((self.portfolio_value * self.max_position_size) / 100)  # Assume $100 min price
        adjusted_size = max(0, min(adjusted_size, max_size))
        
        return adjusted_size
    
    def update_portfolio_value(self, new_value: float):
        """Update the portfolio value for position sizing calculations."""
        self.portfolio_value = new_value
    
    def get_risk_metrics(self, 
                        position_size: int,
                        price: float,
                        stop_loss_pct: float = 0.02) -> Dict[str, float]:
        """
        Calculate risk metrics for a proposed position.
        
        Args:
            position_size: Number of shares
            price: Current price
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Dictionary with risk metrics
        """
        position_value = position_size * price
        max_loss = position_value * stop_loss_pct
        
        return {
            'position_value': position_value,
            'portfolio_weight': position_value / self.portfolio_value,
            'max_loss_amount': max_loss,
            'max_loss_pct': max_loss / self.portfolio_value,
            'risk_reward_ratio': stop_loss_pct / 0.04,  # Assume 4% target
            'position_risk_score': min(position_value / self.portfolio_value / self.max_position_size, 1.0)
        }
