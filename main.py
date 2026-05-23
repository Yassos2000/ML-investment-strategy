import warnings
from pathlib import Path

import pandas as pd

from config import (
    BENCHMARK_TICKER,
    END_DATE,
    HORIZONS,
    KMEANS_RSI_INIT,
    RESULTS_DIR,
    START_DATE,
    TOP_N_LIQUID,
)
from backtest.backtest import run_backtest
from data.data_loader import download_ohlcv, download_spy, get_sp500_tickers
from features.fama_french import calculate_rolling_betas, download_fama_french_factors
from features.indicators import add_technical_indicators
from features.returns import (
    aggregate_monthly_data,
    clip_outliers,
    compute_horizon_returns,
)
from results.performance import calculate_performance_metrics, plot_results
from strategy.clustering import select_high_rsi_cluster

warnings.filterwarnings("ignore")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading S&P 500 tickers...")
    tickers = get_sp500_tickers()

    print("Downloading OHLCV data for S&P 500...")
    daily = download_ohlcv(tickers, START_DATE, END_DATE)
    print(f"After download: daily shape = {getattr(daily, 'shape', None)}, unique tickers = {daily['ticker'].nunique() if 'ticker' in daily.columns else 'N/A'}")

    print("Computing technical indicators...")
    daily = add_technical_indicators(daily)
    # show NaN counts for key feature columns
    feature_cols = ["RSI", "bollinger_band_width", "ATR_zscore", "MACD_zscore", "beta_mkt"]
    nan_counts = {c: int(daily[c].isna().sum()) for c in feature_cols if c in daily.columns}
    print(f"After indicators: daily shape = {daily.shape}, NaN counts = {nan_counts}")

    print("Aggregating to monthly frequency and selecting liquid universe...")
    monthly = aggregate_monthly_data(daily, TOP_N_LIQUID)
    print(f"After monthly agg: monthly shape = {monthly.shape}, unique months = {monthly['date'].nunique()}, unique tickers = {monthly['ticker'].nunique()}")

    print("Computing horizon returns and clipping outliers...")
    monthly = compute_horizon_returns(monthly, HORIZONS)
    monthly = clip_outliers(monthly, [f"return_{h}m" for h in HORIZONS])

    print("Downloading Fama-French 5 factors...")
    factors = download_fama_french_factors(START_DATE, END_DATE)

    print("Calculating rolling factor betas...")
    monthly = calculate_rolling_betas(monthly, factors)

    print("Running clustering and selecting universe...")
    monthly = select_high_rsi_cluster(monthly, KMEANS_RSI_INIT)
    print(f"After clustering: monthly shape = {monthly.shape}, selected count = {int(monthly['selected_universe'].sum())}")

    print(f"Downloading benchmark data for {BENCHMARK_TICKER}...")
    benchmark = download_spy(BENCHMARK_TICKER, START_DATE, END_DATE)

    # Quick optimization sanity check: try optimizing first available period
    try:
        from strategy.optimization import optimize_portfolio
        universe = monthly[monthly["selected_universe"]].copy()
        pivot = universe.pivot(index="date", columns="ticker", values="adjClose").sort_index()
        test_weights_done = False
        for i in range(len(pivot.index) - 1):
            date = pivot.index[i]
            next_date = pivot.index[i + 1]
            cur = pivot.loc[date]
            nxt = pivot.loc[next_date]
            tickers_now = cur.dropna().index.intersection(nxt.dropna().index).tolist()
            history = pivot.loc[:date, tickers_now].dropna(axis=1, how="any")
            if len(tickers_now) >= 2 and history.shape[0] >= 12:
                w = optimize_portfolio(history)
                print(f"Optimization sanity check for {date}: returned {len(w)} weights, sum={w.sum():.6f}")
                print(w.head(20).to_string())
                test_weights_done = True
                break
        if not test_weights_done:
            print("Optimization sanity check: no valid period found to test optimizer")
    except Exception as e:
        print(f"Optimization sanity check failed: {e}")

    print("Running monthly backtest...")
    backtest_df = run_backtest(monthly, benchmark)

    print("Calculating performance metrics...")
    metrics = calculate_performance_metrics(backtest_df)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Print cumulative vs SPY
    try:
        strat_final = backtest_df["strategy_cum"].iloc[-1]
        bench_final = backtest_df["benchmark_cum"].iloc[-1]
        diff = strat_final - bench_final
        print(f"Cumulative Return (Strategy): {strat_final - 1:.4f}")
        print(f"Cumulative Return (SPY): {bench_final - 1:.4f}")
        print(f"Cumulative Return vs SPY: {diff:.4f}")
    except Exception:
        pass

    print("Plotting and saving results...")
    plot_results(backtest_df, RESULTS_DIR / "strategy_performance.png")

    print(f"Backtest complete. Plot saved to {RESULTS_DIR / 'strategy_performance.png'}")


if __name__ == "__main__":
    main()
