import numpy as np
import pandas as pd


def garman_klass_volatility(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 30) -> pd.Series:
    log_hl = np.log(high / low)
    log_co = np.log(close / close.shift(1))
    var = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
    return var.rolling(window=window, min_periods=window).mean().pow(0.5)


def rsi(series: pd.Series, period: int = 20) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(alpha=1 / period, min_periods=period).mean()
    ema_down = down.ewm(alpha=1 / period, min_periods=period).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_band_width(series: pd.Series, period: int = 20) -> pd.Series:
    log_price = np.log1p(series)
    ma = log_price.rolling(period, min_periods=period).mean()
    sd = log_price.rolling(period, min_periods=period).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    return upper - lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def dollar_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    return close * volume


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    indicators = []
    for ticker, group in df.groupby("ticker", sort=False):
        group = group.copy()
        group["gk_volatility"] = garman_klass_volatility(group["high"], group["low"], group["adjClose"], window=30)
        group["RSI"] = rsi(group["adjClose"], period=20)
        group["bollinger_band_width"] = bollinger_band_width(group["adjClose"], period=20)
        group["ATR"] = atr(group["high"], group["low"], group["adjClose"], period=14)
        group["MACD"] = macd(group["adjClose"], fast=12, slow=26)
        group["dollar_volume"] = dollar_volume(group["adjClose"], group["volume"])
        group["ATR_zscore"] = zscore(group["ATR"])
        group["MACD_zscore"] = zscore(group["MACD"])
        indicators.append(group)
    return pd.concat(indicators).reset_index(drop=True)
