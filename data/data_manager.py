"""
Data fetching and management utilities
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import aiohttp
from config.config import CONFIG

class DataManager:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
    
    def fetch_historical_data(self, symbols: List[str], period: str = '2y', 
                            interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        """
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    # Add technical indicators
                    df = self._add_technical_indicators(df)
                    data[symbol] = df
                    self.cache[symbol] = df
                    self.last_update[symbol] = datetime.now()
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    def fetch_real_time_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch real-time data for symbols
        """
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                history = ticker.history(period='1d', interval='1m')
                
                if not history.empty:
                    current_price = history['Close'].iloc[-1]
                    volume = history['Volume'].iloc[-1]
                    
                    data[symbol] = {
                        'price': current_price,
                        'volume': volume,
                        'change': info.get('regularMarketChangePercent', 0),
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                print(f"Error fetching real-time data for {symbol}: {e}")
        
        return data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe
        """
        try:
            import ta
            import pandas_ta as ta_lib
            
            # Price-based indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # Volatility indicators
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self._bollinger_bands(df['Close'])
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            # Momentum indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Support/Resistance levels
            df['Support'] = df['Low'].rolling(window=20).min()
            df['Resistance'] = df['High'].rolling(window=20).max()
            
            # Returns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Volatility
            df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
            
        except ImportError as e:
            print(f"Warning: Some technical indicators not available: {e}")
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a dataframe
        """
        return self._add_technical_indicators(df.copy())
    
    def _bollinger_bands(self, price_series: pd.Series, window: int = 20, 
                        num_std: float = 2) -> tuple:
        """
        Calculate Bollinger Bands
        """
        middle = price_series.rolling(window=window).mean()
        std = price_series.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Get fundamental data for a symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0)
            }
            
            return fundamentals
            
        except Exception as e:
            print(f"Error fetching fundamental data for {symbol}: {e}")
            return {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data
        """
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (beyond 5 standard deviations)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[np.abs(df[col] - mean) <= (5 * std)]
        
        return df
    
    def get_market_regime(self, df: pd.DataFrame) -> str:
        """
        Determine current market regime
        """
        recent_returns = df['Returns'].tail(20)
        volatility = recent_returns.std() * np.sqrt(252)
        trend = df['Close'].tail(20).mean() / df['Close'].tail(60).mean()
        
        if volatility > 0.25:
            return 'high_volatility'
        elif trend > 1.05:
            return 'bull_market'
        elif trend < 0.95:
            return 'bear_market'
        else:
            return 'sideways'
    
    def get_historical_data(self, symbols: List[str], start_date: str = None, 
                           end_date: str = None, period: str = '2y') -> Dict[str, pd.DataFrame]:
        """
        Get historical data for symbols (alias for fetch_historical_data)
        """
        if start_date and end_date:
            # Use specific date range
            data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    
                    if not df.empty:
                        df = self._add_technical_indicators(df)
                        data[symbol] = df
                        
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
            return data
        else:
            return self.fetch_historical_data(symbols, period)

# Global data manager instance
data_manager = DataManager()
