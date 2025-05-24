"""
Risk calculator module for comprehensive risk metrics and assessments.
Calculates VaR, CVaR, correlation risk, drawdown metrics, and other risk measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class RiskCalculator:
    """
    Comprehensive risk calculation and assessment system.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99, 0.999]):
        """
        Initialize the risk calculator.
        
        Args:
            confidence_levels: VaR confidence levels to calculate
        """
        self.confidence_levels = confidence_levels
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_var(self, 
                     returns: pd.Series, 
                     method: str = 'historical',
                     window: int = 252) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using different methods.
        
        Args:
            returns: Return series
            method: 'historical', 'parametric', or 'monte_carlo'
            window: Lookback window for calculations
            
        Returns:
            Dictionary of VaR values for different confidence levels
        """
        if len(returns) < window:
            window = len(returns)
        
        recent_returns = returns.tail(window)
        var_results = {}
        
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            
            if method == 'historical':
                var_value = np.percentile(recent_returns, alpha * 100)
            
            elif method == 'parametric':
                mean = recent_returns.mean()
                std = recent_returns.std()
                var_value = mean + std * stats.norm.ppf(alpha)
            
            elif method == 'monte_carlo':
                var_value = self._monte_carlo_var(recent_returns, alpha)
            
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            var_results[f'var_{conf_level:.3f}'] = var_value
        
        return var_results
    
    def calculate_cvar(self, 
                      returns: pd.Series, 
                      confidence_level: float = 0.95,
                      window: int = 252) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Return series
            confidence_level: Confidence level for CVaR calculation
            window: Lookback window
            
        Returns:
            CVaR value
        """
        if len(returns) < window:
            window = len(returns)
        
        recent_returns = returns.tail(window)
        alpha = 1 - confidence_level
        var_threshold = np.percentile(recent_returns, alpha * 100)
        
        # CVaR is the mean of returns below VaR threshold
        tail_returns = recent_returns[recent_returns <= var_threshold]
        return tail_returns.mean() if len(tail_returns) > 0 else var_threshold
    
    def calculate_drawdown_metrics(self, 
                                 portfolio_value: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive drawdown metrics.
        
        Args:
            portfolio_value: Portfolio value series
            
        Returns:
            Dictionary of drawdown metrics
        """
        # Calculate drawdown series
        rolling_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Maximum drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                    current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if any(drawdown < 0) else 0
        
        # Calmar ratio (Annual return / Max Drawdown)
        if len(portfolio_value) > 1:
            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
            periods_per_year = 252 if len(portfolio_value) > 252 else len(portfolio_value)
            annual_return = (1 + total_return) ** (periods_per_year / len(portfolio_value)) - 1
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            calmar_ratio = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown': avg_drawdown,
            'calmar_ratio': calmar_ratio,
            'current_drawdown': drawdown.iloc[-1]
        }
    
    def calculate_portfolio_risk_metrics(self, 
                                       returns: pd.Series,
                                       benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Basic risk metrics
        metrics['volatility_annual'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        
        # Tail risk metrics
        var_metrics = self.calculate_var(returns)
        metrics.update(var_metrics)
        metrics['cvar_95'] = self.calculate_cvar(returns, 0.95)
        
        # Information ratio and tracking error (if benchmark provided)
        if benchmark_returns is not None:
            aligned_returns, aligned_benchmark = self._align_series(returns, benchmark_returns)
            excess_returns = aligned_returns - aligned_benchmark
            metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
            metrics['information_ratio'] = (excess_returns.mean() / excess_returns.std() * np.sqrt(252) 
                                          if excess_returns.std() != 0 else 0)
            metrics['beta'] = self._calculate_beta(aligned_returns, aligned_benchmark)
            metrics['alpha'] = self._calculate_alpha(aligned_returns, aligned_benchmark, metrics['beta'])
        
        return metrics
    
    def calculate_correlation_risk(self, 
                                 returns_matrix: pd.DataFrame,
                                 threshold: float = 0.7) -> Dict[str, Any]:
        """
        Analyze correlation risk across strategies/assets.
        
        Args:
            returns_matrix: DataFrame with returns for different strategies/assets
            threshold: Correlation threshold for risk identification
            
        Returns:
            Dictionary with correlation risk analysis
        """
        correlation_matrix = returns_matrix.corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_correlations.append({
                        'asset1': correlation_matrix.columns[i],
                        'asset2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'risk_level': 'HIGH' if abs(corr_value) > 0.8 else 'MEDIUM'
                    })
        
        # Calculate concentration risk
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-8]  # Remove near-zero eigenvalues
        
        # Effective number of independent bets
        effective_bets = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        
        # Diversification ratio
        portfolio_weights = np.ones(len(returns_matrix.columns)) / len(returns_matrix.columns)
        individual_vols = returns_matrix.std()
        weighted_avg_vol = np.sum(portfolio_weights * individual_vols)
        portfolio_vol = np.sqrt(np.dot(portfolio_weights, np.dot(correlation_matrix * 
                                                               np.outer(individual_vols, individual_vols), 
                                                               portfolio_weights)))
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations,
            'effective_bets': effective_bets,
            'diversification_ratio': diversification_ratio,
            'concentration_risk': 'HIGH' if effective_bets < len(returns_matrix.columns) * 0.5 else 'LOW'
        }
    
    def calculate_leverage_risk(self, 
                              positions: pd.DataFrame,
                              portfolio_value: float,
                              max_leverage: float = 2.0) -> Dict[str, Any]:
        """
        Calculate leverage-related risk metrics.
        
        Args:
            positions: DataFrame with position data
            portfolio_value: Current portfolio value
            max_leverage: Maximum allowed leverage
            
        Returns:
            Dictionary with leverage risk metrics
        """
        # Calculate gross and net exposure
        gross_exposure = positions['market_value'].abs().sum()
        net_exposure = positions['market_value'].sum()
        
        # Calculate leverage ratios
        gross_leverage = gross_exposure / portfolio_value if portfolio_value > 0 else 0
        net_leverage = abs(net_exposure) / portfolio_value if portfolio_value > 0 else 0
        
        # Long/short exposure
        long_exposure = positions[positions['market_value'] > 0]['market_value'].sum()
        short_exposure = abs(positions[positions['market_value'] < 0]['market_value'].sum())
        
        return {
            'gross_leverage': gross_leverage,
            'net_leverage': net_leverage,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'leverage_utilization': gross_leverage / max_leverage,
            'leverage_risk': 'HIGH' if gross_leverage > max_leverage * 0.8 else 'MEDIUM' if gross_leverage > max_leverage * 0.6 else 'LOW'
        }
    
    def calculate_liquidity_risk(self, 
                               positions: pd.DataFrame,
                               avg_volumes: Dict[str, float],
                               participation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Assess liquidity risk for current positions.
        
        Args:
            positions: Current positions DataFrame
            avg_volumes: Dictionary of average daily volumes for each symbol
            participation_rate: Maximum participation rate in daily volume
            
        Returns:
            Dictionary with liquidity risk assessment
        """
        liquidity_metrics = {}
        
        for _, position in positions.iterrows():
            symbol = position['symbol']
            position_size = abs(position['quantity'])
            
            if symbol in avg_volumes:
                daily_volume = avg_volumes[symbol]
                max_daily_trade = daily_volume * participation_rate
                days_to_liquidate = position_size / max_daily_trade if max_daily_trade > 0 else float('inf')
                
                liquidity_metrics[symbol] = {
                    'days_to_liquidate': days_to_liquidate,
                    'liquidity_risk': ('HIGH' if days_to_liquidate > 5 else 
                                     'MEDIUM' if days_to_liquidate > 2 else 'LOW')
                }
        
        # Overall portfolio liquidity risk
        high_risk_positions = sum(1 for metrics in liquidity_metrics.values() 
                                if metrics['liquidity_risk'] == 'HIGH')
        total_positions = len(liquidity_metrics)
        
        overall_risk = ('HIGH' if high_risk_positions / total_positions > 0.3 else
                       'MEDIUM' if high_risk_positions / total_positions > 0.1 else 'LOW')
        
        return {
            'position_liquidity': liquidity_metrics,
            'overall_liquidity_risk': overall_risk,
            'high_risk_positions': high_risk_positions,
            'total_positions': total_positions
        }
    
    def stress_test_portfolio(self, 
                            returns_matrix: pd.DataFrame,
                            scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Perform stress testing on portfolio under various scenarios.
        
        Args:
            returns_matrix: Historical returns matrix
            scenarios: Dictionary of stress scenarios
            
        Returns:
            Dictionary with scenario impacts
        """
        portfolio_weights = np.ones(len(returns_matrix.columns)) / len(returns_matrix.columns)
        base_portfolio_return = np.mean(returns_matrix.dot(portfolio_weights))
        
        scenario_results = {}
        
        for scenario_name, shocks in scenarios.items():
            shocked_returns = returns_matrix.copy()
            
            # Apply shocks to relevant assets
            for asset, shock in shocks.items():
                if asset in shocked_returns.columns:
                    shocked_returns[asset] = shocked_returns[asset] + shock
            
            # Calculate portfolio impact
            shocked_portfolio_return = np.mean(shocked_returns.dot(portfolio_weights))
            impact = (shocked_portfolio_return - base_portfolio_return) * 100
            
            scenario_results[scenario_name] = impact
        
        return scenario_results
    
    def _monte_carlo_var(self, returns: pd.Series, alpha: float, 
                        num_simulations: int = 10000) -> float:
        """Monte Carlo simulation for VaR calculation."""
        mean = returns.mean()
        std = returns.std()
        
        simulated_returns = np.random.normal(mean, std, num_simulations)
        return np.percentile(simulated_returns, alpha * 100)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (return/downside deviation)."""
        excess_returns = returns - self.risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        if downside_deviation == 0:
            return 0
        
        return excess_returns.mean() * np.sqrt(252) / downside_deviation
    
    def _align_series(self, series1: pd.Series, 
                     series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two time series on common dates."""
        common_dates = series1.index.intersection(series2.index)
        return series1.loc[common_dates], series2.loc[common_dates]
    
    def _calculate_beta(self, returns: pd.Series, 
                       benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        aligned_returns, aligned_benchmark = self._align_series(returns, benchmark_returns)
        
        if len(aligned_returns) < 2:
            return 1.0
        
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        return covariance / benchmark_variance if benchmark_variance != 0 else 1.0
    
    def _calculate_alpha(self, returns: pd.Series, 
                        benchmark_returns: pd.Series, beta: float) -> float:
        """Calculate alpha (excess return over CAPM prediction)."""
        aligned_returns, aligned_benchmark = self._align_series(returns, benchmark_returns)
        
        portfolio_return = aligned_returns.mean() * 252
        benchmark_return = aligned_benchmark.mean() * 252
        expected_return = self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate)
        
        return portfolio_return - expected_return
