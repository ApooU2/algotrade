"""
Risk monitoring module for real-time risk assessment and alerting.
Monitors portfolio risk limits, generates alerts, and provides risk dashboards.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    timestamp: datetime
    risk_type: str
    level: RiskLevel
    message: str
    current_value: float
    threshold: float
    recommendations: List[str]


class RiskMonitor:
    """
    Real-time risk monitoring and alerting system.
    """
    
    def __init__(self, 
                 risk_limits: Dict[str, float],
                 alert_callback: Optional[Callable] = None):
        """
        Initialize the risk monitor.
        
        Args:
            risk_limits: Dictionary of risk limits
            alert_callback: Optional callback function for alerts
        """
        self.risk_limits = risk_limits
        self.alert_callback = alert_callback
        self.alerts_history = []
        self.last_check = datetime.now()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Risk monitoring flags
        self.monitoring_enabled = True
        self.alert_cooldown = {}  # Prevent spam alerts
        
        # Default risk limits if not provided
        self.default_limits = {
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.1,
            'max_drawdown': 0.15,
            'max_leverage': 2.0,
            'min_sharpe_ratio': 0.5,
            'max_var_95': 0.05,
            'max_correlation': 0.7,
            'max_concentration': 0.3
        }
        
        # Update with provided limits
        for key, value in self.default_limits.items():
            if key not in self.risk_limits:
                self.risk_limits[key] = value
    
    def check_portfolio_risk(self, 
                           portfolio_data: Dict[str, Any],
                           positions: pd.DataFrame,
                           returns: pd.Series) -> List[RiskAlert]:
        """
        Comprehensive portfolio risk check.
        
        Args:
            portfolio_data: Portfolio performance data
            positions: Current positions DataFrame
            returns: Portfolio returns series
            
        Returns:
            List of risk alerts
        """
        alerts = []
        current_time = datetime.now()
        
        # Check drawdown risk
        alerts.extend(self._check_drawdown_risk(portfolio_data, current_time))
        
        # Check position concentration risk
        alerts.extend(self._check_concentration_risk(positions, current_time))
        
        # Check leverage risk
        alerts.extend(self._check_leverage_risk(positions, portfolio_data.get('portfolio_value', 0), current_time))
        
        # Check performance risk
        alerts.extend(self._check_performance_risk(returns, current_time))
        
        # Check correlation risk
        alerts.extend(self._check_correlation_risk(positions, current_time))
        
        # Check VaR risk
        alerts.extend(self._check_var_risk(returns, current_time))
        
        # Update alerts history
        self.alerts_history.extend(alerts)
        
        # Trigger callbacks for new alerts
        for alert in alerts:
            if self.alert_callback and self._should_trigger_alert(alert):
                self.alert_callback(alert)
        
        self.last_check = current_time
        return alerts
    
    def check_position_risk(self, 
                          symbol: str,
                          position_size: int,
                          price: float,
                          portfolio_value: float) -> List[RiskAlert]:
        """
        Check risk for a specific position.
        
        Args:
            symbol: Asset symbol
            position_size: Number of shares
            price: Current price
            portfolio_value: Total portfolio value
            
        Returns:
            List of risk alerts
        """
        alerts = []
        current_time = datetime.now()
        
        position_value = position_size * price
        position_weight = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Check position size limit
        if position_weight > self.risk_limits['max_position_size']:
            alert = RiskAlert(
                timestamp=current_time,
                risk_type='POSITION_SIZE',
                level=RiskLevel.HIGH,
                message=f"Position {symbol} exceeds size limit: {position_weight:.2%} > {self.risk_limits['max_position_size']:.2%}",
                current_value=position_weight,
                threshold=self.risk_limits['max_position_size'],
                recommendations=[
                    f"Reduce {symbol} position size",
                    "Diversify into other assets",
                    "Review position sizing strategy"
                ]
            )
            alerts.append(alert)
        
        return alerts
    
    def check_real_time_risk(self, 
                           market_data: Dict[str, Any],
                           portfolio_metrics: Dict[str, float]) -> List[RiskAlert]:
        """
        Real-time risk monitoring during trading hours.
        
        Args:
            market_data: Current market data
            portfolio_metrics: Real-time portfolio metrics
            
        Returns:
            List of risk alerts
        """
        alerts = []
        current_time = datetime.now()
        
        # Check market volatility spike
        if 'market_volatility' in market_data:
            vol = market_data['market_volatility']
            if vol > 0.5:  # 50% volatility
                alert = RiskAlert(
                    timestamp=current_time,
                    risk_type='MARKET_VOLATILITY',
                    level=RiskLevel.HIGH,
                    message=f"Extreme market volatility detected: {vol:.1%}",
                    current_value=vol,
                    threshold=0.5,
                    recommendations=[
                        "Reduce position sizes",
                        "Increase stop losses",
                        "Consider defensive strategies"
                    ]
                )
                alerts.append(alert)
        
        # Check intraday drawdown
        if 'intraday_drawdown' in portfolio_metrics:
            dd = abs(portfolio_metrics['intraday_drawdown'])
            if dd > 0.05:  # 5% intraday drawdown
                alert = RiskAlert(
                    timestamp=current_time,
                    risk_type='INTRADAY_DRAWDOWN',
                    level=RiskLevel.CRITICAL if dd > 0.1 else RiskLevel.HIGH,
                    message=f"Large intraday drawdown: {dd:.2%}",
                    current_value=dd,
                    threshold=0.05,
                    recommendations=[
                        "Review open positions",
                        "Consider closing losing positions",
                        "Halt new position entries"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def generate_risk_report(self, 
                           portfolio_data: Dict[str, Any],
                           positions: pd.DataFrame,
                           timeframe: str = '1D') -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_data: Portfolio data
            positions: Current positions
            timeframe: Report timeframe
            
        Returns:
            Risk report dictionary
        """
        # Get recent alerts
        cutoff_time = datetime.now() - timedelta(days=1 if timeframe == '1D' else 7)
        recent_alerts = [alert for alert in self.alerts_history 
                        if alert.timestamp >= cutoff_time]
        
        # Risk summary
        risk_summary = {
            'total_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a.level == RiskLevel.CRITICAL]),
            'high_alerts': len([a for a in recent_alerts if a.level == RiskLevel.HIGH]),
            'medium_alerts': len([a for a in recent_alerts if a.level == RiskLevel.MEDIUM]),
            'alert_categories': {}
        }
        
        # Categorize alerts
        for alert in recent_alerts:
            if alert.risk_type not in risk_summary['alert_categories']:
                risk_summary['alert_categories'][alert.risk_type] = 0
            risk_summary['alert_categories'][alert.risk_type] += 1
        
        # Current risk status
        current_risk_level = self._calculate_overall_risk_level(recent_alerts)
        
        # Risk metrics compliance
        compliance_status = {}
        if positions is not None and not positions.empty:
            # Position concentration
            total_value = positions['market_value'].abs().sum()
            max_position = positions['market_value'].abs().max()
            concentration = max_position / total_value if total_value > 0 else 0
            compliance_status['concentration'] = concentration <= self.risk_limits['max_concentration']
            
            # Leverage
            portfolio_value = portfolio_data.get('portfolio_value', 0)
            leverage = total_value / portfolio_value if portfolio_value > 0 else 0
            compliance_status['leverage'] = leverage <= self.risk_limits['max_leverage']
        
        return {
            'timestamp': datetime.now(),
            'timeframe': timeframe,
            'overall_risk_level': current_risk_level,
            'risk_summary': risk_summary,
            'compliance_status': compliance_status,
            'recent_alerts': recent_alerts,
            'recommendations': self._generate_risk_recommendations(recent_alerts)
        }
    
    def set_dynamic_limits(self, 
                          market_conditions: Dict[str, float],
                          performance_metrics: Dict[str, float]):
        """
        Dynamically adjust risk limits based on market conditions.
        
        Args:
            market_conditions: Current market conditions
            performance_metrics: Recent performance metrics
        """
        # Adjust based on market volatility
        if 'market_volatility' in market_conditions:
            vol = market_conditions['market_volatility']
            if vol > 0.3:  # High volatility
                self.risk_limits['max_position_size'] *= 0.8
                self.risk_limits['max_portfolio_risk'] *= 0.7
            elif vol < 0.1:  # Low volatility
                self.risk_limits['max_position_size'] *= 1.1
                self.risk_limits['max_portfolio_risk'] *= 1.2
        
        # Adjust based on recent performance
        if 'recent_sharpe' in performance_metrics:
            sharpe = performance_metrics['recent_sharpe']
            if sharpe < 0.5:  # Poor performance
                self.risk_limits['max_position_size'] *= 0.9
                self.risk_limits['max_portfolio_risk'] *= 0.8
        
        # Ensure limits stay within reasonable bounds
        self.risk_limits['max_position_size'] = max(0.05, min(0.2, self.risk_limits['max_position_size']))
        self.risk_limits['max_portfolio_risk'] = max(0.01, min(0.05, self.risk_limits['max_portfolio_risk']))
    
    def _check_drawdown_risk(self, 
                           portfolio_data: Dict[str, Any],
                           timestamp: datetime) -> List[RiskAlert]:
        """Check drawdown risk."""
        alerts = []
        
        if 'current_drawdown' in portfolio_data:
            drawdown = abs(portfolio_data['current_drawdown'])
            
            if drawdown > self.risk_limits['max_drawdown']:
                level = RiskLevel.CRITICAL if drawdown > 0.2 else RiskLevel.HIGH
                alert = RiskAlert(
                    timestamp=timestamp,
                    risk_type='DRAWDOWN',
                    level=level,
                    message=f"Drawdown exceeds limit: {drawdown:.2%} > {self.risk_limits['max_drawdown']:.2%}",
                    current_value=drawdown,
                    threshold=self.risk_limits['max_drawdown'],
                    recommendations=[
                        "Review and close losing positions",
                        "Reduce overall exposure",
                        "Implement stricter stop losses"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_concentration_risk(self, 
                                positions: pd.DataFrame,
                                timestamp: datetime) -> List[RiskAlert]:
        """Check position concentration risk."""
        alerts = []
        
        if positions is not None and not positions.empty:
            total_value = positions['market_value'].abs().sum()
            if total_value > 0:
                max_position = positions['market_value'].abs().max()
                concentration = max_position / total_value
                
                if concentration > self.risk_limits['max_concentration']:
                    alert = RiskAlert(
                        timestamp=timestamp,
                        risk_type='CONCENTRATION',
                        level=RiskLevel.MEDIUM if concentration < 0.4 else RiskLevel.HIGH,
                        message=f"Portfolio concentration too high: {concentration:.2%} > {self.risk_limits['max_concentration']:.2%}",
                        current_value=concentration,
                        threshold=self.risk_limits['max_concentration'],
                        recommendations=[
                            "Diversify portfolio holdings",
                            "Reduce largest positions",
                            "Add uncorrelated assets"
                        ]
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _check_leverage_risk(self, 
                           positions: pd.DataFrame,
                           portfolio_value: float,
                           timestamp: datetime) -> List[RiskAlert]:
        """Check leverage risk."""
        alerts = []
        
        if positions is not None and not positions.empty and portfolio_value > 0:
            gross_exposure = positions['market_value'].abs().sum()
            leverage = gross_exposure / portfolio_value
            
            if leverage > self.risk_limits['max_leverage']:
                alert = RiskAlert(
                    timestamp=timestamp,
                    risk_type='LEVERAGE',
                    level=RiskLevel.HIGH if leverage > 3.0 else RiskLevel.MEDIUM,
                    message=f"Leverage exceeds limit: {leverage:.2f}x > {self.risk_limits['max_leverage']:.2f}x",
                    current_value=leverage,
                    threshold=self.risk_limits['max_leverage'],
                    recommendations=[
                        "Reduce position sizes",
                        "Close some positions",
                        "Increase cash allocation"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_performance_risk(self, 
                              returns: pd.Series,
                              timestamp: datetime) -> List[RiskAlert]:
        """Check performance-related risks."""
        alerts = []
        
        if len(returns) > 30:  # Need sufficient data
            recent_returns = returns.tail(30)
            sharpe_ratio = (recent_returns.mean() / recent_returns.std() * np.sqrt(252) 
                          if recent_returns.std() > 0 else 0)
            
            if sharpe_ratio < self.risk_limits['min_sharpe_ratio']:
                alert = RiskAlert(
                    timestamp=timestamp,
                    risk_type='PERFORMANCE',
                    level=RiskLevel.MEDIUM,
                    message=f"Low Sharpe ratio: {sharpe_ratio:.2f} < {self.risk_limits['min_sharpe_ratio']:.2f}",
                    current_value=sharpe_ratio,
                    threshold=self.risk_limits['min_sharpe_ratio'],
                    recommendations=[
                        "Review trading strategies",
                        "Optimize risk-return profile",
                        "Consider strategy adjustments"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_correlation_risk(self, 
                              positions: pd.DataFrame,
                              timestamp: datetime) -> List[RiskAlert]:
        """Check correlation risk (placeholder - would need correlation data)."""
        alerts = []
        # This would require correlation matrix - simplified implementation
        return alerts
    
    def _check_var_risk(self, 
                      returns: pd.Series,
                      timestamp: datetime) -> List[RiskAlert]:
        """Check Value at Risk."""
        alerts = []
        
        if len(returns) > 50:
            var_95 = abs(np.percentile(returns, 5))
            
            if var_95 > self.risk_limits['max_var_95']:
                alert = RiskAlert(
                    timestamp=timestamp,
                    risk_type='VAR',
                    level=RiskLevel.MEDIUM,
                    message=f"VaR exceeds limit: {var_95:.3f} > {self.risk_limits['max_var_95']:.3f}",
                    current_value=var_95,
                    threshold=self.risk_limits['max_var_95'],
                    recommendations=[
                        "Reduce portfolio volatility",
                        "Diversify risk exposure",
                        "Implement hedging strategies"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _should_trigger_alert(self, alert: RiskAlert) -> bool:
        """Check if alert should be triggered (avoid spam)."""
        cooldown_key = f"{alert.risk_type}_{alert.level.value}"
        current_time = datetime.now()
        
        if cooldown_key in self.alert_cooldown:
            time_since_last = current_time - self.alert_cooldown[cooldown_key]
            cooldown_period = timedelta(minutes=30)  # 30-minute cooldown
            
            if time_since_last < cooldown_period:
                return False
        
        self.alert_cooldown[cooldown_key] = current_time
        return True
    
    def _calculate_overall_risk_level(self, alerts: List[RiskAlert]) -> RiskLevel:
        """Calculate overall risk level from alerts."""
        if not alerts:
            return RiskLevel.LOW
        
        critical_count = len([a for a in alerts if a.level == RiskLevel.CRITICAL])
        high_count = len([a for a in alerts if a.level == RiskLevel.HIGH])
        
        if critical_count > 0:
            return RiskLevel.CRITICAL
        elif high_count >= 2:
            return RiskLevel.HIGH
        elif high_count >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_risk_recommendations(self, alerts: List[RiskAlert]) -> List[str]:
        """Generate risk management recommendations."""
        if not alerts:
            return ["Risk levels are within acceptable limits"]
        
        recommendations = set()
        
        # Add recommendations from alerts
        for alert in alerts:
            recommendations.update(alert.recommendations)
        
        # Add general recommendations based on alert patterns
        risk_types = [alert.risk_type for alert in alerts]
        
        if risk_types.count('DRAWDOWN') > 0:
            recommendations.add("Consider implementing circuit breakers")
        
        if risk_types.count('CONCENTRATION') > 0:
            recommendations.add("Increase portfolio diversification")
        
        if len(set(risk_types)) > 3:  # Multiple risk types
            recommendations.add("Conduct comprehensive risk review")
        
        return list(recommendations)
