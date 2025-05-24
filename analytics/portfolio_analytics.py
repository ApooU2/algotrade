"""
Advanced Portfolio Analytics
Comprehensive performance tracking and analysis tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    alpha: float
    beta: float
    information_ratio: float
    treynor_ratio: float
    var_95: float
    cvar_95: float

class AdvancedPortfolioAnalytics:
    """Advanced portfolio analytics and performance measurement"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trading_data.db')
        else:
            self.db_path = db_path
        
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure required database tables exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolio history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                daily_return REAL,
                cumulative_return REAL,
                drawdown REAL,
                benchmark_return REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trade history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_date TIMESTAMP NOT NULL,
                exit_date TIMESTAMP,
                strategy TEXT,
                pnl REAL,
                commission REAL,
                slippage REAL,
                hold_period INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                strategy TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_portfolio_snapshot(self, date: datetime, portfolio_value: float, 
                                 cash: float, positions_value: float, 
                                 daily_return: float = 0.0, benchmark_return: float = 0.0):
        """Record portfolio snapshot to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate cumulative return
        cursor.execute('''
            SELECT portfolio_value FROM portfolio_history 
            ORDER BY date DESC LIMIT 1
        ''')
        
        result = cursor.fetchone()
        if result:
            initial_value = result[0]
            cumulative_return = (portfolio_value - initial_value) / initial_value
        else:
            cumulative_return = 0.0
        
        # Calculate drawdown
        cursor.execute('''
            SELECT MAX(portfolio_value) FROM portfolio_history 
            WHERE date <= ?
        ''', (date.strftime('%Y-%m-%d'),))
        
        result = cursor.fetchone()
        if result and result[0]:
            peak_value = result[0]
            drawdown = (peak_value - portfolio_value) / peak_value
        else:
            drawdown = 0.0
        
        cursor.execute('''
            INSERT INTO portfolio_history 
            (date, portfolio_value, cash, positions_value, daily_return, 
             cumulative_return, drawdown, benchmark_return)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date.strftime('%Y-%m-%d'), portfolio_value, cash, positions_value,
              daily_return, cumulative_return, drawdown, benchmark_return))
        
        conn.commit()
        conn.close()
    
    def record_trade(self, symbol: str, side: str, quantity: float, 
                    entry_price: float, entry_date: datetime, strategy: str = None,
                    exit_price: float = None, exit_date: datetime = None,
                    commission: float = 0.0, slippage: float = 0.0):
        """Record trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate P&L if exit price provided
        pnl = None
        hold_period = None
        
        if exit_price and exit_date:
            if side.lower() == 'buy':
                pnl = (exit_price - entry_price) * quantity - commission - slippage
            else:  # sell/short
                pnl = (entry_price - exit_price) * quantity - commission - slippage
            
            hold_period = (exit_date - entry_date).days
        
        cursor.execute('''
            INSERT INTO trade_history 
            (symbol, side, quantity, entry_price, exit_price, entry_date, exit_date,
             strategy, pnl, commission, slippage, hold_period)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, side, quantity, entry_price, exit_price, 
              entry_date.isoformat(), exit_date.isoformat() if exit_date else None,
              strategy, pnl, commission, slippage, hold_period))
        
        conn.commit()
        conn.close()
    
    def get_portfolio_history(self, start_date: datetime = None, 
                             end_date: datetime = None) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM portfolio_history"
        params = []
        
        if start_date or end_date:
            query += " WHERE"
            conditions = []
            
            if start_date:
                conditions.append(" date >= ?")
                params.append(start_date.strftime('%Y-%m-%d'))
            
            if end_date:
                conditions.append(" date <= ?")
                params.append(end_date.strftime('%Y-%m-%d'))
            
            query += " AND".join(conditions)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    def get_trade_history(self, start_date: datetime = None, 
                         end_date: datetime = None) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trade_history"
        params = []
        
        if start_date or end_date:
            query += " WHERE"
            conditions = []
            
            if start_date:
                conditions.append(" entry_date >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append(" entry_date <= ?")
                params.append(end_date.isoformat())
            
            query += " AND".join(conditions)
        
        query += " ORDER BY entry_date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])
        
        return df
    
    def calculate_comprehensive_metrics(self, start_date: datetime = None, 
                                      end_date: datetime = None,
                                      benchmark_returns: pd.Series = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        portfolio_history = self.get_portfolio_history(start_date, end_date)
        trade_history = self.get_trade_history(start_date, end_date)
        
        if portfolio_history.empty:
            raise ValueError("No portfolio history data available")
        
        returns = portfolio_history['daily_return'].dropna()
        portfolio_values = portfolio_history['portfolio_value']
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
        annualized_return = (1 + total_return) ** (365 / len(portfolio_values)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown metrics
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Trade-based metrics
        if not trade_history.empty:
            completed_trades = trade_history.dropna(subset=['pnl'])
            
            if len(completed_trades) > 0:
                winning_trades = completed_trades[completed_trades['pnl'] > 0]
                losing_trades = completed_trades[completed_trades['pnl'] < 0]
                
                win_rate = len(winning_trades) / len(completed_trades)
                average_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                average_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
                largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
                largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
                
                total_wins = winning_trades['pnl'].sum()
                total_losses = abs(losing_trades['pnl'].sum())
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                # Consecutive wins/losses
                consecutive_wins = self._calculate_max_consecutive(completed_trades['pnl'] > 0)
                consecutive_losses = self._calculate_max_consecutive(completed_trades['pnl'] < 0)
            else:
                win_rate = average_win = average_loss = largest_win = largest_loss = 0
                profit_factor = consecutive_wins = consecutive_losses = 0
        else:
            win_rate = average_win = average_loss = largest_win = largest_loss = 0
            profit_factor = consecutive_wins = consecutive_losses = 0
        
        # Alpha and Beta (if benchmark provided)
        alpha = beta = information_ratio = treynor_ratio = 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            covariance = np.cov(returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            benchmark_return = benchmark_returns.mean() * 252
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
            
            # Information ratio
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            # Treynor ratio
            treynor_ratio = (annualized_return - risk_free_rate) / beta if beta != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _calculate_max_consecutive(self, condition_series: pd.Series) -> int:
        """Calculate maximum consecutive occurrences"""
        if condition_series.empty:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in condition_series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def generate_performance_report(self, start_date: datetime = None, 
                                  end_date: datetime = None,
                                  save_path: str = None) -> str:
        """Generate comprehensive performance report"""
        metrics = self.calculate_comprehensive_metrics(start_date, end_date)
        
        report = f"""
COMPREHENSIVE PORTFOLIO PERFORMANCE REPORT
==========================================

Report Period: {start_date or 'Inception'} to {end_date or 'Present'}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RETURN METRICS
--------------
Total Return:                {metrics.total_return:.2%}
Annualized Return:           {metrics.annualized_return:.2%}
Volatility (Annualized):     {metrics.volatility:.2%}

RISK-ADJUSTED METRICS
---------------------
Sharpe Ratio:                {metrics.sharpe_ratio:.3f}
Sortino Ratio:               {metrics.sortino_ratio:.3f}
Calmar Ratio:                {metrics.calmar_ratio:.3f}
Maximum Drawdown:            {metrics.max_drawdown:.2%}

BENCHMARK COMPARISON
--------------------
Alpha:                       {metrics.alpha:.2%}
Beta:                        {metrics.beta:.3f}
Information Ratio:           {metrics.information_ratio:.3f}
Treynor Ratio:               {metrics.treynor_ratio:.3f}

TRADING METRICS
---------------
Win Rate:                    {metrics.win_rate:.1%}
Profit Factor:               {metrics.profit_factor:.2f}
Average Winning Trade:       ${metrics.average_win:.2f}
Average Losing Trade:        ${metrics.average_loss:.2f}
Largest Winning Trade:       ${metrics.largest_win:.2f}
Largest Losing Trade:        ${metrics.largest_loss:.2f}
Max Consecutive Wins:        {metrics.consecutive_wins}
Max Consecutive Losses:      {metrics.consecutive_losses}

RISK METRICS
------------
95% Value at Risk (Daily):   {metrics.var_95:.2%}
95% CVaR (Daily):            {metrics.cvar_95:.2%}

PERFORMANCE ANALYSIS
--------------------
"""
        
        # Add performance analysis
        if metrics.sharpe_ratio > 1.0:
            report += "• Excellent risk-adjusted returns (Sharpe > 1.0)\n"
        elif metrics.sharpe_ratio > 0.5:
            report += "• Good risk-adjusted returns (Sharpe > 0.5)\n"
        else:
            report += "• Poor risk-adjusted returns (Sharpe < 0.5)\n"
        
        if metrics.max_drawdown > -0.20:
            report += "• High drawdown risk (Max DD > 20%)\n"
        elif metrics.max_drawdown > -0.10:
            report += "• Moderate drawdown risk (Max DD > 10%)\n"
        else:
            report += "• Low drawdown risk (Max DD < 10%)\n"
        
        if metrics.win_rate > 0.6:
            report += "• High win rate (> 60%)\n"
        elif metrics.win_rate > 0.4:
            report += "• Moderate win rate (40-60%)\n"
        else:
            report += "• Low win rate (< 40%)\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def create_performance_dashboard(self, save_path: str = None):
        """Create comprehensive performance dashboard with charts"""
        portfolio_history = self.get_portfolio_history()
        
        if portfolio_history.empty:
            print("No data available for dashboard creation")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_history.index, portfolio_history['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        axes[0, 1].plot(portfolio_history.index, portfolio_history['cumulative_return'] * 100)
        axes[0, 1].set_title('Cumulative Returns (%)')
        axes[0, 1].set_ylabel('Cumulative Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Drawdown
        axes[1, 0].fill_between(portfolio_history.index, 
                               portfolio_history['drawdown'] * 100, 
                               0, alpha=0.7, color='red')
        axes[1, 0].set_title('Drawdown (%)')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Daily returns distribution
        returns = portfolio_history['daily_return'].dropna()
        axes[1, 1].hist(returns, bins=50, alpha=0.7)
        axes[1, 1].axvline(returns.mean(), color='red', linestyle='--', label='Mean')
        axes[1, 1].axvline(returns.mean() + returns.std(), color='orange', linestyle='--', label='+1 Std')
        axes[1, 1].axvline(returns.mean() - returns.std(), color='orange', linestyle='--', label='-1 Std')
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Global analytics instance
analytics = AdvancedPortfolioAnalytics()
