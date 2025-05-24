"""
Helper functions and utilities for the algorithmic trading bot.
Common calculations, data processing, and utility functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log' returns
        
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change().dropna()
    elif method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if returns.std() == 0:
        return 0
    
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_max_drawdown(portfolio_value: pd.Series) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.
    
    Args:
        portfolio_value: Portfolio value series
        
    Returns:
        Dictionary with drawdown metrics
    """
    rolling_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    
    max_drawdown = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Find drawdown duration
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
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_date': max_dd_date,
        'max_drawdown_duration': max_drawdown_duration,
        'current_drawdown': drawdown.iloc[-1]
    }


def calculate_volatility(returns: pd.Series, 
                        window: Optional[int] = None,
                        annualize: bool = True) -> Union[float, pd.Series]:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        returns: Returns series
        window: Rolling window size (None for full period)
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility value or series
    """
    if window is None:
        vol = returns.std()
    else:
        vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def normalize_data(data: pd.DataFrame, 
                  method: str = 'zscore') -> pd.DataFrame:
    """
    Normalize data using various methods.
    
    Args:
        data: Data to normalize
        method: 'zscore', 'minmax', or 'robust'
        
    Returns:
        Normalized data
    """
    if method == 'zscore':
        return (data - data.mean()) / data.std()
    elif method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'robust':
        median = data.median()
        mad = (data - median).abs().median()
        return (data - median) / mad
    else:
        raise ValueError("Method must be 'zscore', 'minmax', or 'robust'")


def validate_data(data: pd.DataFrame, 
                 required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate data quality and completeness.
    
    Args:
        data: Data to validate
        required_columns: List of required columns
        
    Returns:
        Validation report
    """
    report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'missing_columns': [],
        'missing_data': {},
        'data_types': {},
        'date_range': {}
    }
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        report['missing_columns'] = missing_cols
        report['errors'].append(f"Missing required columns: {missing_cols}")
        report['is_valid'] = False
    
    # Check for missing data
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(data) * 100
            report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            if missing_pct > 5:  # More than 5% missing
                report['warnings'].append(f"Column {col} has {missing_pct:.1f}% missing data")
    
    # Check data types
    for col in data.columns:
        report['data_types'][col] = str(data[col].dtype)
    
    # Check date range if index is datetime
    if isinstance(data.index, pd.DatetimeIndex):
        report['date_range'] = {
            'start': data.index.min(),
            'end': data.index.max(),
            'duration_days': (data.index.max() - data.index.min()).days
        }
    
    return report


def resample_data(data: pd.DataFrame, 
                 freq: str,
                 agg_method: str = 'last') -> pd.DataFrame:
    """
    Resample time series data to different frequency.
    
    Args:
        data: Time series data
        freq: Target frequency ('1D', '1H', '5min', etc.)
        agg_method: Aggregation method ('last', 'first', 'mean', 'ohlc')
        
    Returns:
        Resampled data
    """
    if agg_method == 'ohlc':
        # Special handling for OHLC data
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    else:
        resampled = data.resample(freq).agg(agg_method)
    
    return resampled.dropna()


def calculate_correlation_matrix(returns_data: pd.DataFrame,
                               method: str = 'pearson',
                               window: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix for returns data.
    
    Args:
        returns_data: Returns DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')
        window: Rolling window size (None for full period)
        
    Returns:
        Correlation matrix
    """
    if window is None:
        return returns_data.corr(method=method)
    else:
        return returns_data.rolling(window=window).corr(method=method)


def detect_outliers(data: pd.Series, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in data series.
    
    Args:
        data: Data series
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")


def calculate_performance_metrics(returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
    metrics['volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate)
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        metrics['downside_deviation'] = downside_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['downside_deviation']
    else:
        metrics['downside_deviation'] = 0
        metrics['sortino_ratio'] = 0
    
    # Win/loss metrics
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    metrics['win_rate'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    metrics['avg_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
    metrics['avg_loss'] = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
    metrics['profit_factor'] = (metrics['avg_win'] * len(positive_returns)) / (metrics['avg_loss'] * len(negative_returns)) if metrics['avg_loss'] > 0 else 0
    
    # Additional metrics
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    metrics['var_95'] = returns.quantile(0.05)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean() if any(returns <= metrics['var_95']) else metrics['var_95']
    
    # Benchmark comparison
    if benchmark_returns is not None:
        aligned_returns, aligned_benchmark = align_series(returns, benchmark_returns)
        excess_returns = aligned_returns - aligned_benchmark
        
        metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
        metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Beta calculation
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Alpha calculation
        benchmark_return = aligned_benchmark.mean() * 252
        expected_return = risk_free_rate + metrics['beta'] * (benchmark_return - risk_free_rate)
        metrics['alpha'] = metrics['annualized_return'] - expected_return
    
    return metrics


def align_series(*series: pd.Series) -> Tuple[pd.Series, ...]:
    """
    Align multiple time series on common dates.
    
    Args:
        *series: Variable number of pandas Series
        
    Returns:
        Tuple of aligned series
    """
    if len(series) < 2:
        return series
    
    # Find common dates
    common_index = series[0].index
    for s in series[1:]:
        common_index = common_index.intersection(s.index)
    
    # Return aligned series
    return tuple(s.loc[common_index] for s in series)


def generate_date_range(start_date: Union[str, datetime],
                       end_date: Union[str, datetime],
                       freq: str = 'D',
                       market_only: bool = True) -> pd.DatetimeIndex:
    """
    Generate date range for backtesting or analysis.
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D', 'B', 'H', etc.)
        market_only: Whether to include only market days
        
    Returns:
        DatetimeIndex
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    if market_only and freq in ['D', 'B']:
        # Use business days for daily frequency
        return pd.bdate_range(start=start_date, end=end_date, freq='B')
    else:
        return pd.date_range(start=start_date, end=end_date, freq=freq)


def format_number(value: float, 
                 format_type: str = 'percentage',
                 decimals: int = 2) -> str:
    """
    Format numbers for display.
    
    Args:
        value: Value to format
        format_type: 'percentage', 'currency', 'number'
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return 'N/A'
    
    if format_type == 'percentage':
        return f"{value * 100:.{decimals}f}%"
    elif format_type == 'currency':
        return f"${value:,.{decimals}f}"
    elif format_type == 'number':
        return f"{value:,.{decimals}f}"
    else:
        return str(value)


def create_bins(data: pd.Series, 
               method: str = 'equal_width',
               n_bins: int = 10) -> Tuple[pd.Series, np.ndarray]:
    """
    Create bins for data analysis.
    
    Args:
        data: Data to bin
        method: Binning method ('equal_width', 'equal_freq', 'quantile')
        n_bins: Number of bins
        
    Returns:
        Tuple of (binned_data, bin_edges)
    """
    if method == 'equal_width':
        binned_data, bin_edges = pd.cut(data, bins=n_bins, retbins=True)
    elif method == 'equal_freq':
        binned_data, bin_edges = pd.qcut(data, q=n_bins, retbins=True, duplicates='drop')
    elif method == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = data.quantile(quantiles).values
        binned_data = pd.cut(data, bins=bin_edges, include_lowest=True)
    else:
        raise ValueError("Method must be 'equal_width', 'equal_freq', or 'quantile'")
    
    return binned_data, bin_edges


def safe_divide(numerator: Union[float, pd.Series],
               denominator: Union[float, pd.Series],
               default_value: float = 0.0) -> Union[float, pd.Series]:
    """
    Perform safe division avoiding division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default_value: Value to return when denominator is zero
        
    Returns:
        Division result or default value
    """
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(default_value)
    else:
        return numerator / denominator if denominator != 0 else default_value
