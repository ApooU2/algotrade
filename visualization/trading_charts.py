"""
Trading charts module for technical analysis visualization.
Provides candlestick charts, indicators, and signal overlays.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import mplfinance as mpf


class TradingCharts:
    """
    Advanced trading charts with technical indicators and signal visualization.
    """
    
    def __init__(self):
        """Initialize the trading charts system."""
        self.colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'volume': '#26a69a',
            'signal_buy': '#4caf50',
            'signal_sell': '#f44336',
            'indicator1': '#2196f3',
            'indicator2': '#ff9800',
            'indicator3': '#9c27b0'
        }
    
    def plot_candlestick_with_indicators(self, 
                                       data: pd.DataFrame,
                                       indicators: Dict[str, pd.Series],
                                       signals: Optional[pd.DataFrame] = None,
                                       title: str = "Price Chart with Indicators") -> go.Figure:
        """
        Create interactive candlestick chart with technical indicators.
        
        Args:
            data: OHLCV data
            indicators: Dictionary of indicator series
            signals: Buy/sell signals DataFrame
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Create subplots
        indicator_count = len(indicators)
        subplot_rows = 2 + (indicator_count > 0)
        
        row_heights = [0.6, 0.2]
        if indicator_count > 0:
            row_heights.append(0.2)
        
        subplot_titles = ['Price', 'Volume']
        if indicator_count > 0:
            subplot_titles.append('Technical Indicators')
        
        fig = make_subplots(
            rows=subplot_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        # Add signals if provided
        if signals is not None:
            if 'buy_signals' in signals.columns:
                buy_signals = signals[signals['buy_signals']].index
                if len(buy_signals) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals,
                            y=data.loc[buy_signals, 'low'] * 0.98,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color=self.colors['signal_buy']
                            ),
                            name='Buy Signal'
                        ),
                        row=1, col=1
                    )
            
            if 'sell_signals' in signals.columns:
                sell_signals = signals[signals['sell_signals']].index
                if len(sell_signals) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals,
                            y=data.loc[sell_signals, 'high'] * 1.02,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color=self.colors['signal_sell']
                            ),
                            name='Sell Signal'
                        ),
                        row=1, col=1
                    )
        
        # Volume chart
        volume_colors = [self.colors['bullish'] if close >= open_ 
                        else self.colors['bearish'] 
                        for close, open_ in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                marker_color=volume_colors,
                name='Volume',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Technical indicators
        if indicator_count > 0:
            colors = [self.colors['indicator1'], self.colors['indicator2'], 
                     self.colors['indicator3']]
            
            for i, (name, series) in enumerate(indicators.items()):
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        name=name,
                        line=dict(color=colors[i % len(colors)]),
                        opacity=0.8
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600 + (indicator_count > 0) * 200,
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            showlegend=True
        )
        
        # Update x-axis
        fig.update_xaxes(
            type='date',
            rangeslider_visible=False
        )
        
        return fig
    
    def plot_strategy_signals(self, 
                            data: pd.DataFrame,
                            strategy_name: str,
                            entry_signals: pd.Series,
                            exit_signals: pd.Series,
                            positions: pd.Series) -> go.Figure:
        """
        Plot strategy signals and positions over price chart.
        
        Args:
            data: OHLCV data
            strategy_name: Name of the strategy
            entry_signals: Entry signal series
            exit_signals: Exit signal series
            positions: Position series (1 for long, -1 for short, 0 for flat)
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=[f'{strategy_name} - Price & Signals', 'Position']
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                name='Close Price',
                line=dict(color='#1f77b4')
            ),
            row=1, col=1
        )
        
        # Entry signals
        entry_points = entry_signals[entry_signals == 1].index
        if len(entry_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=entry_points,
                    y=data.loc[entry_points, 'close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color=self.colors['signal_buy']
                    ),
                    name='Entry Signal'
                ),
                row=1, col=1
            )
        
        # Exit signals
        exit_points = exit_signals[exit_signals == 1].index
        if len(exit_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=exit_points,
                    y=data.loc[exit_points, 'close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=8,
                        color=self.colors['signal_sell']
                    ),
                    name='Exit Signal'
                ),
                row=1, col=1
            )
        
        # Position chart
        position_colors = np.where(positions > 0, self.colors['bullish'],
                                 np.where(positions < 0, self.colors['bearish'], 'gray'))
        
        fig.add_trace(
            go.Bar(
                x=positions.index,
                y=positions.values,
                marker_color=position_colors,
                name='Position',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{strategy_name} Trading Signals',
            height=600,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def plot_correlation_heatmap(self, returns_data: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap for strategy returns.
        
        Args:
            returns_data: DataFrame with strategy returns
            
        Returns:
            Plotly figure
        """
        correlation_matrix = returns_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Strategy Returns Correlation Matrix',
            width=600,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def plot_drawdown_underwater(self, portfolio_value: pd.Series) -> go.Figure:
        """
        Create underwater equity curve (drawdown visualization).
        
        Args:
            portfolio_value: Portfolio value series
            
        Returns:
            Plotly figure
        """
        # Calculate drawdown
        rolling_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        
        fig = go.Figure()
        
        # Underwater chart
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                fill='tonexty',
                line=dict(color='rgba(214, 39, 40, 0.8)'),
                fillcolor='rgba(214, 39, 40, 0.3)',
                name='Drawdown %'
            )
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title='Underwater Equity Curve (Drawdown)',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_rolling_metrics(self, 
                           returns: pd.Series,
                           window: int = 252) -> go.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Daily returns series
            window: Rolling window size
            
        Returns:
            Plotly figure
        """
        # Calculate rolling metrics
        rolling_return = returns.rolling(window).mean() * 252
        rolling_volatility = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_volatility
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=['Rolling Annual Return', 'Rolling Volatility', 'Rolling Sharpe Ratio']
        )
        
        # Rolling return
        fig.add_trace(
            go.Scatter(
                x=rolling_return.index,
                y=rolling_return * 100,
                name='Rolling Return (%)',
                line=dict(color='#2196f3')
            ),
            row=1, col=1
        )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_volatility.index,
                y=rolling_volatility * 100,
                name='Rolling Volatility (%)',
                line=dict(color='#ff9800')
            ),
            row=2, col=1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                name='Rolling Sharpe',
                line=dict(color='#4caf50')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'Rolling Performance Metrics ({window}-day window)',
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_mplfinance_chart(self, 
                              data: pd.DataFrame,
                              indicators: Optional[Dict] = None,
                              signals: Optional[pd.DataFrame] = None,
                              style: str = 'charles') -> None:
        """
        Create traditional candlestick chart using mplfinance.
        
        Args:
            data: OHLCV data
            indicators: Dictionary of indicators to overlay
            signals: Buy/sell signals
            style: Chart style
        """
        # Prepare additional plots
        addplot = []
        
        if indicators:
            for name, series in indicators.items():
                addplot.append(
                    mpf.make_addplot(series, color='blue', alpha=0.7)
                )
        
        if signals is not None:
            if 'buy_signals' in signals.columns:
                buy_points = signals['buy_signals']
                addplot.append(
                    mpf.make_addplot(buy_points * data['low'], 
                                   type='scatter', markersize=100, 
                                   marker='^', color='green')
                )
            
            if 'sell_signals' in signals.columns:
                sell_points = signals['sell_signals']
                addplot.append(
                    mpf.make_addplot(sell_points * data['high'], 
                                   type='scatter', markersize=100, 
                                   marker='v', color='red')
                )
        
        # Create chart
        mpf.plot(data, type='candle', style=style, volume=True,
                addplot=addplot if addplot else None,
                title='Trading Chart',
                figsize=(12, 8))
        
        plt.show()
