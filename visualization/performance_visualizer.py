"""
Performance visualization module for the algorithmic trading bot.
Creates comprehensive charts and reports for strategy performance analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import io
import base64


class PerformanceVisualizer:
    """
    Comprehensive performance visualization system for trading strategies.
    """
    
    def __init__(self, style: str = 'darkgrid'):
        """
        Initialize the performance visualizer.
        
        Args:
            style: Seaborn style theme
        """
        plt.style.use('seaborn-v0_8')
        sns.set_style(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_portfolio_performance(self, portfolio_data: pd.DataFrame, 
                                 benchmark_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create an interactive portfolio performance chart.
        
        Args:
            portfolio_data: DataFrame with portfolio value over time
            benchmark_data: Optional benchmark comparison data
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Value', 'Daily Returns', 'Drawdown'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['portfolio_value'],
                name='Portfolio',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        if benchmark_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_data['value'],
                    name='Benchmark',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Daily returns
        returns = portfolio_data['portfolio_value'].pct_change()
        fig.add_trace(
            go.Bar(
                x=portfolio_data.index,
                y=returns,
                name='Daily Returns',
                marker_color=np.where(returns >= 0, '#2ca02c', '#d62728')
            ),
            row=2, col=1
        )
        
        # Drawdown
        rolling_max = portfolio_data['portfolio_value'].expanding().max()
        drawdown = (portfolio_data['portfolio_value'] - rolling_max) / rolling_max
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_data.index,
                y=drawdown * 100,
                name='Drawdown %',
                fill='tonexty',
                line=dict(color='#d62728'),
                fillcolor='rgba(214, 39, 40, 0.3)'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Portfolio Performance Analysis',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_strategy_comparison(self, strategy_results: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Compare performance of multiple strategies.
        
        Args:
            strategy_results: Dictionary of strategy names and their performance data
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Returns', 'Return Distribution', 
                          'Rolling Sharpe Ratio', 'Monthly Returns Heatmap'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Cumulative returns
        for i, (strategy, data) in enumerate(strategy_results.items()):
            cumulative_returns = (1 + data['returns']).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=cumulative_returns,
                    name=strategy,
                    line=dict(color=self.colors[i % len(self.colors)])
                ),
                row=1, col=1
            )
        
        # Return distribution
        for i, (strategy, data) in enumerate(strategy_results.items()):
            fig.add_trace(
                go.Histogram(
                    x=data['returns'],
                    name=f'{strategy} Returns',
                    opacity=0.7,
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=1, col=2
            )
        
        # Rolling Sharpe ratio
        for i, (strategy, data) in enumerate(strategy_results.items()):
            rolling_sharpe = data['returns'].rolling(window=252).mean() / data['returns'].rolling(window=252).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rolling_sharpe,
                    name=f'{strategy} Sharpe',
                    line=dict(color=self.colors[i % len(self.colors)])
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Strategy Performance Comparison',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_trade_analysis(self, trades_df: pd.DataFrame) -> go.Figure:
        """
        Analyze individual trades performance.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('P&L Distribution', 'Win/Loss Ratio', 
                          'Trade Duration', 'Monthly Trade Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # P&L Distribution
        fig.add_trace(
            go.Histogram(
                x=trades_df['pnl'],
                name='P&L Distribution',
                marker_color='#1f77b4',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Win/Loss pie chart
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] <= 0])
        
        fig.add_trace(
            go.Pie(
                labels=['Wins', 'Losses'],
                values=[wins, losses],
                marker_colors=['#2ca02c', '#d62728']
            ),
            row=1, col=2
        )
        
        # Trade duration
        fig.add_trace(
            go.Histogram(
                x=trades_df['duration_hours'],
                name='Duration (hours)',
                marker_color='#ff7f0e',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Monthly trade count
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        monthly_counts = trades_df.groupby('month').size()
        
        fig.add_trace(
            go.Bar(
                x=monthly_counts.index.astype(str),
                y=monthly_counts.values,
                name='Monthly Trades',
                marker_color='#9467bd'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Trade Analysis Dashboard',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_performance_report(self, portfolio_data: pd.DataFrame, 
                                trades_df: pd.DataFrame,
                                strategy_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive HTML performance report.
        
        Args:
            portfolio_data: Portfolio performance data
            trades_df: Individual trades data
            strategy_results: Strategy performance metrics
            
        Returns:
            HTML report string
        """
        # Calculate key metrics
        total_return = (portfolio_data['portfolio_value'].iloc[-1] / 
                       portfolio_data['portfolio_value'].iloc[0] - 1) * 100
        
        daily_returns = portfolio_data['portfolio_value'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        max_drawdown = ((portfolio_data['portfolio_value'] / 
                        portfolio_data['portfolio_value'].expanding().max()) - 1).min() * 100
        
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
        
        # Generate charts
        portfolio_fig = self.plot_portfolio_performance(portfolio_data)
        trade_fig = self.plot_trade_analysis(trades_df)
        
        # Convert plots to HTML
        portfolio_html = portfolio_fig.to_html(include_plotlyjs='cdn')
        trade_html = trade_fig.to_html(include_plotlyjs='cdn')
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Algorithmic Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          background-color: #f0f0f0; border-radius: 5px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; 
                          text-align: center; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Algorithmic Trading Performance Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Key Performance Metrics</h2>
                <div class="metric">
                    <h3>Total Return</h3>
                    <p>{total_return:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Sharpe Ratio</h3>
                    <p>{sharpe_ratio:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Max Drawdown</h3>
                    <p>{max_drawdown:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Win Rate</h3>
                    <p>{win_rate:.1f}%</p>
                </div>
                <div class="metric">
                    <h3>Total Trades</h3>
                    <p>{len(trades_df)}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Portfolio Performance</h2>
                {portfolio_html}
            </div>
            
            <div class="section">
                <h2>Trade Analysis</h2>
                {trade_html}
            </div>
            
            <div class="section">
                <h2>Strategy Breakdown</h2>
                <table border="1" style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <th>Strategy</th>
                        <th>Total Return (%)</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown (%)</th>
                        <th>Win Rate (%)</th>
                    </tr>
        """
        
        for strategy, metrics in strategy_results.items():
            html_report += f"""
                    <tr>
                        <td>{strategy}</td>
                        <td>{metrics.get('total_return', 0):.2f}</td>
                        <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                        <td>{metrics.get('max_drawdown', 0):.2f}</td>
                        <td>{metrics.get('win_rate', 0):.1f}</td>
                    </tr>
            """
        
        html_report += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_report
    
    def save_report(self, html_content: str, filepath: str):
        """
        Save HTML report to file.
        
        Args:
            html_content: HTML report content
            filepath: Path to save the report
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
