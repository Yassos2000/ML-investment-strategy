import pandas as pd

from strategy.optimization import optimize_portfolio


def run_backtest(monthly_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = monthly_df.copy()
    monthly_df["date"] = pd.to_datetime(monthly_df["date"]).dt.to_period("M").dt.to_timestamp("M", how="end")
    benchmark_df = benchmark_df.copy()
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"]).dt.to_period("M").dt.to_timestamp("M", how="end")
    benchmark_monthly = (
        benchmark_df.groupby("date")["adjClose"].last().pct_change().reset_index()
    )
    # build a full price pivot for all tickers (so history is continuous)
    price_pivot = monthly_df.pivot(index="date", columns="ticker", values="adjClose")
    price_pivot = price_pivot.sort_index()
    # selection lookup (which tickers are selected each date)
    selection = monthly_df[monthly_df["selected_universe"]][["date", "ticker"]]
    selection = selection.set_index(["date", "ticker"]).sort_index()
    results = []
    weights_history = []
    dates = sorted(price_pivot.index)
    for i in range(len(dates) - 1):
        date = dates[i]
        next_date = dates[i + 1]
        current_prices = price_pivot.loc[date]
        next_prices = price_pivot.loc[next_date]
        # tickers selected this date (ensure they have price entries)
        selected_rows = selection.loc[date] if (date in selection.index.get_level_values(0)) else None
        if selected_rows is None:
            tickers = []
        else:
            tickers = list(selected_rows.index.intersection(current_prices.dropna().index).intersection(next_prices.dropna().index))
        # Debug: how many stocks are available this month
        print(f"[Backtest] Period {date} -> {next_date}: available tickers {len(tickers)}")
        if len(tickers) < 2:
            print(f"[Backtest] skipping period {date} -> {next_date}: less than 2 tickers")
            continue
        # Use only the most recent 12 months of history ending at `date`.
        history_full = price_pivot.loc[:date, tickers]
        history = history_full.tail(12).dropna(axis=1, how="any")
        print(f"[Backtest] history rows for {date}: full_rows={history_full.shape[0]} tail12_rows={history.shape[0]} columns_after_drop={history.shape[1]}")
        if history.shape[0] < 12 or history.shape[1] < 2:
            print(f"[Backtest] skipping period {date} -> {next_date}: insufficient usable history (rows={history.shape[0]}, cols={history.shape[1]})")
            continue
        weights = optimize_portfolio(history)
        # Debug: print raw weights returned by optimizer
        try:
            print(f"[Backtest] raw optimizer weights (sum={weights.sum():.6f}):")
            print(weights.to_string())
        except Exception:
            pass
        if weights.sum() <= 0:
            print(f"[Backtest] optimizer returned zero/invalid weights for {date}")
            # still record zero weights for debugging
        weights = weights.reindex(tickers).fillna(0.0)
        returns = next_prices[tickers] / current_prices[tickers] - 1
        # Debug: print individual returns used to compute portfolio return
        try:
            print(f"[Backtest] Returns for period {date} -> {next_date}:")
            for t in tickers:
                r = returns.get(t, float('nan'))
                print(f"  {t}: {r:.4f}")
        except Exception:
            pass

        portfolio_return = (weights * returns).sum()
        # Debug: print monthly portfolio weights
        try:
            print(f"[Backtest] Portfolio weights for {date} -> {next_date}:")
            for t, w in weights.reindex(tickers).fillna(0.0).items():
                print(f"  {t}: {w:.4f}")
        except Exception:
            pass

        print(f"[Backtest] portfolio return for {date} -> {next_date}: {portfolio_return:.6f}")
        results.append({"date": next_date, "strategy_return": portfolio_return})
        weights_history.append({"date": date, **weights.to_dict()})
    result_df = pd.DataFrame(results, columns=["date", "strategy_return"])
    if result_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "strategy_return",
                "benchmark_return",
                "strategy_cum",
                "benchmark_cum",
            ]
        )
    result_df = result_df.sort_values("date").reset_index(drop=True)
    result_df = result_df.merge(
        benchmark_monthly.rename(columns={"adjClose": "benchmark_return"}),
        on="date",
        how="left",
    )
    result_df["benchmark_return"] = result_df["benchmark_return"].fillna(0.0)
    result_df["strategy_cum"] = (1 + result_df["strategy_return"]).cumprod()
    result_df["benchmark_cum"] = (1 + result_df["benchmark_return"]).cumprod()
    return result_df
