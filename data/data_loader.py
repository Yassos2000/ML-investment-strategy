import io
import math
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
import yfinance as yf


DATA_CACHE_DIR = Path(__file__).parent / "cache"


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _retry_request(func, max_retries: int = 3, backoff_factor: float = 1.0):
    """Generic retry wrapper with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff_factor * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)


def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace('.', '-', regex=False).tolist()
    return sorted(set(tickers))


def _normalize_ticker_data(ticker: str, raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    if raw.empty:
        return raw
    if "Adj Close" not in raw.columns:
        return raw
    raw = raw.rename(columns={"Adj Close": "adjClose"})
    raw = raw.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    raw["ticker"] = ticker
    raw.index.name = "date"
    return raw[["ticker", "open", "high", "low", "close", "adjClose", "volume"]]


def _download_chunk(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    def _download():
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            threads=False,
            progress=False,
        )
        return raw

    raw = _retry_request(_download, max_retries=3, backoff_factor=1.0)
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.levels[0]:
                continue
            ticker_data = raw.loc[:, (ticker, slice(None))].copy()
            ticker_data.columns = ticker_data.columns.droplevel(0)
            ticker_data = _normalize_ticker_data(ticker, ticker_data)
            frames.append(ticker_data)
    else:
        ticker = tickers[0]
        ticker_data = _normalize_ticker_data(ticker, raw)
        frames.append(ticker_data)
    if frames:
        return pd.concat(frames)
    return pd.DataFrame(columns=["ticker", "open", "high", "low", "close", "adjClose", "volume"])


def download_ohlcv(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    _ensure_cache_dir()
    
    # Check for cached data
    cache_file = DATA_CACHE_DIR / f"ohlcv_{start}_{end}.csv"
    if cache_file.exists():
        print(f"Loading cached OHLCV data from {cache_file.name}...")
        data = pd.read_csv(cache_file)
        return data
    
    chunks = []
    chunk_size = 20
    total = len(tickers)
    for i in range(0, len(tickers), chunk_size):
        subset = tickers[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = ((total - 1) // chunk_size) + 1
        print(
            f"Downloading chunk {chunk_num} of {total_chunks} ({len(subset)} tickers)"
        )
        chunk = _download_chunk(subset, start, end)
        if not chunk.empty:
            chunks.append(chunk)
    
    if not chunks:
        return pd.DataFrame(
            columns=["ticker", "open", "high", "low", "close", "adjClose", "volume"]
        )
    
    data = pd.concat(chunks)
    data = data.reset_index()
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Save to cache
    data.to_csv(cache_file, index=False)
    print(f"Saved OHLCV data to cache: {cache_file.name}")
    
    return data


def download_spy(symbol: str, start: str, end: str) -> pd.DataFrame:
    _ensure_cache_dir()
    
    # Check for cached data
    cache_file = DATA_CACHE_DIR / f"{symbol}_{start}_{end}.csv"
    if cache_file.exists():
        print(f"Loading cached {symbol} data from {cache_file.name}...")
        return pd.read_csv(cache_file)
    
    def _download():
        return yf.download(
            symbol, start=start, end=end, auto_adjust=False, progress=False
        )

    df = _retry_request(_download, max_retries=3, backoff_factor=1.0)
    
    if df.empty:
        return df
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df = df.rename(columns={"Adj Close": "adjClose"})
    df.index.name = "date"
    df = df[["Open", "High", "Low", "Close", "adjClose", "Volume"]].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df.reset_index()
    
    # Save to cache
    df.to_csv(cache_file, index=False)
    print(f"Saved {symbol} data to cache: {cache_file.name}")
    
    return df
