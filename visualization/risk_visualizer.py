"""
Risk visualization module for comprehensive risk analysis and monitoring.
Provides risk metrics visualization, VaR analysis, and risk monitoring dashboards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta


class RiskVisualizer:
    """
    Comprehensive risk visualization and analysis system.
    """
    
    def __init__(self):
        """Initialize the risk visualizer."""
        self.colors = {
            'low_risk': '#4caf50',
            'medium_risk': '#ff9800',
            'high_risk': '#f44336',
            'critical_risk': '#9c27b0'
        }
    
    def plot_var_analysis(self, 
                         returns: pd.Series,
                         confidence_levels: List[float] = [0.95, 0.99, 0.999]) -> go.Figure:
        """
        Create Value at Risk (VaR) analysis visualization.
        
        Args:
            returns: Daily returns series
            confidence_levels: VaR confidence levels
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Return Distribution', 'VaR Levels', 
                          'Rolling VaR (95%)', 'Tail Risk Analysis'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Return distribution with VaR levels
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Return Distribution',
                opacity=0.7,
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Add VaR lines
        colors = ['red', 'orange', 'purple']
        for i, conf_level in enumerate(confidence_levels):
            var_value = np.percentile(returns, (1 - conf_level) * 100)
            fig.add_vline(
                x=var_value,
                line_dash="dash",
                line_color=colors[i],
                annotation_text=f"VaR {conf_level:.1%}: {var_value:.3f}",
                row=1, col=1
            )
        
        # VaR levels bar chart
        var_values = [np.percentile(returns, (1 - cl) * 100) for cl in confidence_levels]
        fig.add_trace(
            go.Bar(
                x=[f"{cl:.1%}" for cl in confidence_levels],
                y=np.abs(var_values),
                name='VaR Levels',
                marker_color=['red', 'orange', 'purple']
            ),
            row=1, col=2
        )
        
        # Rolling VaR (95%)
        rolling_var = returns.rolling(window=252).quantile(0.05)
        fig.add_trace(
            go.Scatter(
                x=rolling_var.index,
                y=rolling_var,
                name='Rolling VaR (95%)',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Tail risk analysis (returns below VaR)
        var_95 = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_95]
        
        fig.add_trace(
            go.Histogram(
                x=tail_returns,
                name='Tail Returns (below VaR)',
                marker_color='darkred',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Value at Risk (VaR) Analysis',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_risk_metrics_dashboard(self, 
                                  portfolio_data: pd.DataFrame,
                                  benchmark_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create comprehensive risk metrics dashboard.
        
        Args:
            portfolio_data: Portfolio performance data
            benchmark_data: Optional benchmark data
            
        Returns:
            Plotly figure
        """
        returns = portfolio_data['portfolio_value'].pct_change().dropna()
        
        # Calculate risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_data['portfolio_value'])
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Risk Metrics Overview', 'Return Distribution Stats',
                          'Drawdown Analysis', 'Rolling Volatility',
                          'Beta Analysis', 'Risk-Return Scatter'],
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Risk metrics indicators
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=volatility * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Annual Volatility (%)"},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "lightgray"},
                        {'range': [15, 25], 'color': "yellow"},
                        {'range': [25, 50], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 25
                    }
                }
            ),
            row=1, col=1
        )
        
        # Distribution statistics
        stats_data = {
            'Metric': ['Volatility (%)', 'Sharpe Ratio', 'Skewness', 'Kurtosis', 'Max DD (%)'],
            'Value': [volatility * 100, sharpe_ratio, skewness, kurtosis, max_drawdown * 100]
        }
        
        colors = ['green' if v > 0 else 'red' for v in stats_data['Value']]
        
        fig.add_trace(
            go.Bar(
                x=stats_data['Metric'],
                y=stats_data['Value'],
                marker_color=colors,
                name='Risk Statistics'
            ),
            row=1, col=2
        )
        
        # Drawdown analysis
        drawdown_series = self._calculate_drawdown_series(portfolio_data['portfolio_value'])
        fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series * 100,
                fill='tonexty',
                name='Drawdown (%)',
                line=dict(color='red'),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ),
            row=2, col=1
        )
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol * 100,
                name='30-Day Rolling Volatility (%)',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        
        # Beta analysis (if benchmark provided)
        if benchmark_data is not None:
            benchmark_returns = benchmark_data['value'].pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(benchmark_returns.index)
            aligned_returns = returns.loc[common_dates]
            aligned_benchmark = benchmark_returns.loc[common_dates]
            
            # Calculate rolling beta
            rolling_beta = pd.Series(index=aligned_returns.index, dtype=float)
            for i in range(30, len(aligned_returns)):
                y = aligned_returns.iloc[i-30:i]
                x = aligned_benchmark.iloc[i-30:i]
                beta, _ = np.polyfit(x, y, 1)
                rolling_beta.iloc[i] = beta
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_beta.index,
                    y=rolling_beta,
                    name='Rolling Beta (30-day)',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
        
        # Risk-return scatter
        if len(returns) > 252:
            annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
            fig.add_trace(
                go.Scatter(
                    x=[volatility * 100],
                    y=[annual_return * 100],
                    mode='markers',
                    marker=dict(size=15, color='blue'),
                    name='Portfolio Risk-Return',
                    text=['Portfolio'],
                    textposition='top center'
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title='Risk Metrics Dashboard',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_correlation_risk(self, 
                            returns_matrix: pd.DataFrame,
                            risk_threshold: float = 0.7) -> go.Figure:
        """
        Analyze correlation risk across strategies/assets.
        
        Args:
            returns_matrix: DataFrame with returns for different strategies/assets
            risk_threshold: Correlation threshold for risk warning
            
        Returns:
            Plotly figure
        """
        correlation_matrix = returns_matrix.corr()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Correlation Heatmap', 'High Correlation Warnings'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 8},
                name='Correlation'
            ),
            row=1, col=1
        )
        
        # High correlation warnings
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > risk_threshold:
                    pair_name = f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}"
                    high_corr_pairs.append((pair_name, corr_value))
        
        if high_corr_pairs:
            pair_names, corr_values = zip(*high_corr_pairs)
            colors = ['red' if abs(v) > 0.8 else 'orange' for v in corr_values]
            
            fig.add_trace(
                go.Bar(
                    x=list(pair_names),
                    y=list(corr_values),
                    marker_color=colors,
                    name='High Correlations'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Correlation Risk Analysis',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_stress_test_scenarios(self, 
                                 portfolio_value: pd.Series,
                                 scenarios: Dict[str, float]) -> go.Figure:
        """
        Visualize stress test scenarios impact on portfolio.
        
        Args:
            portfolio_value: Portfolio value series
            scenarios: Dictionary of scenario names and impact percentages
            
        Returns:
            Plotly figure
        """
        current_value = portfolio_value.iloc[-1]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Stress Test Impact', 'Portfolio Value Under Scenarios'],
            row_heights=[0.4, 0.6]
        )
        
        # Stress test impact bar chart
        scenario_names = list(scenarios.keys())
        impacts = list(scenarios.values())
        colors = ['red' if impact < -10 else 'orange' if impact < -5 else 'yellow' 
                 for impact in impacts]
        
        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=impacts,
                marker_color=colors,
                name='Impact (%)',
                text=[f"{impact:.1f}%" for impact in impacts],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Portfolio value under scenarios
        scenario_values = [current_value * (1 + impact/100) for impact in impacts]
        
        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=scenario_values,
                marker_color=colors,
                name='Portfolio Value',
                text=[f"${value:,.0f}" for value in scenario_values],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Add current value line
        fig.add_hline(
            y=current_value,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Current: ${current_value:,.0f}",
            row=2, col=1
        )
        
        fig.update_layout(
            title='Stress Test Analysis',
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def _calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_drawdown_series(self, portfolio_value: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        rolling_max = portfolio_value.expanding().max()
        return (portfolio_value - rolling_max) / rolling_max
    
    def create_risk_report(self, 
                          portfolio_data: pd.DataFrame,
                          returns_matrix: pd.DataFrame,
                          risk_limits: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive risk assessment report.
        
        Args:
            portfolio_data: Portfolio performance data
            returns_matrix: Returns for different strategies
            risk_limits: Dictionary of risk limits
            
        Returns:
            Risk report dictionary
        """
        returns = portfolio_data['portfolio_value'].pct_change().dropna()
        
        # Calculate risk metrics
        metrics = {
            'volatility': returns.std() * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'max_drawdown': self._calculate_max_drawdown(portfolio_data['portfolio_value']),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
        
        # Risk limit violations
        violations = {}
        for limit_name, limit_value in risk_limits.items():
            if limit_name in metrics:
                if limit_name == 'max_drawdown':
                    violations[limit_name] = abs(metrics[limit_name]) > limit_value
                else:
                    violations[limit_name] = metrics[limit_name] > limit_value
        
        # Correlation analysis
        correlation_matrix = returns_matrix.corr()
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_correlations.append({
                        'pair': f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}",
                        'correlation': corr_value
                    })
        
        return {
            'timestamp': datetime.now(),
            'risk_metrics': metrics,
            'limit_violations': violations,
            'high_correlations': high_correlations,
            'overall_risk_score': self._calculate_risk_score(metrics, violations)
        }
    
    def _calculate_risk_score(self, metrics: Dict[str, float], 
                            violations: Dict[str, bool]) -> str:
        """Calculate overall risk score."""
        violation_count = sum(violations.values())
        
        if violation_count == 0:
            return "LOW"
        elif violation_count <= 2:
            return "MEDIUM"
        else:
            return "HIGH"
