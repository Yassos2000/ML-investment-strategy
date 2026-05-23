import pandas as pd


def aggregate_monthly_data(df: pd.DataFrame, top_n: int = 150) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    df["dollar_volume"] = df["adjClose"] * df["volume"]
    df["dv_5y_avg"] = df.groupby("ticker")["dollar_volume"].transform(
        lambda x: x.rolling(window=252 * 5, min_periods=252).mean()
    )
    # Set date as index for pd.Grouper to work with freq="ME"
    df = df.set_index("date")
    monthly = (
        df.groupby(["ticker", pd.Grouper(freq="ME")])
        .last()
        .reset_index()
        .sort_values(["date", "ticker"])
    )
    monthly["universe_rank"] = monthly.groupby("date")["dv_5y_avg"].rank(method="first", ascending=False)
    monthly = monthly[monthly["universe_rank"] <= top_n].copy()
    return monthly.drop(columns=["universe_rank"])


def compute_horizon_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    df["monthly_return"] = df.groupby("ticker")["adjClose"].pct_change(1)
    for horizon in horizons:
        df[f"return_{horizon}m"] = (
            df.groupby("ticker")["adjClose"].shift(-horizon) / df["adjClose"] - 1
        )
    return df


def clip_outliers(df: pd.DataFrame, columns: list[str], lower_quantile: float = 0.005, upper_quantile: float = 0.995) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            continue
        low = df[column].quantile(lower_quantile)
        high = df[column].quantile(upper_quantile)
        df[column] = df[column].clip(lower=low, upper=high)
    return df
