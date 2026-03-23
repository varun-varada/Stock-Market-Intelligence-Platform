"""
Data Acquisition Module
Fetches stock data and news from various sources.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta


def fetch_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)
    
    Returns:
        DataFrame with OHLCV data
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    return df


def fetch_stock_info(symbol: str) -> dict:
    """Fetch stock metadata."""
    stock = yf.Ticker(symbol)
    return stock.info


def fetch_multiple_stocks(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch data for multiple stocks.
    Returns DataFrame with Close prices for all symbols.
    """
    data = {}
    for symbol in symbols:
        df = fetch_stock_data(symbol, period, interval)
        data[symbol] = df['Close']
    
    return pd.DataFrame(data)


def get_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from price data."""
    return df.pct_change().dropna()


def get_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative returns."""
    return (1 + df.pct_change()).cumprod() - 1


# Example usage
if __name__ == "__main__":
    # Test data fetching
    df = fetch_stock_data("AAPL", period="1mo")
    print(f"Fetched {len(df)} rows for AAPL")
    print(df.tail())
