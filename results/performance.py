import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def calculate_performance_metrics(backtest_df: pd.DataFrame) -> dict:
    returns = backtest_df["strategy_return"].dropna()
    months = len(returns)
    if months == 0:
        return {
            "Annual Return": 0.0,
            "Annual Volatility": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Cumulative Return": 0.0,
        }
    cumulative = (1 + returns).prod()
    annual_return = cumulative ** (12.0 / months) - 1
    annual_volatility = returns.std(ddof=1) * (12 ** 0.5)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    cumulative_series = (1 + returns).cumprod()
    drawdown = cumulative_series / cumulative_series.cummax() - 1
    max_drawdown = drawdown.min()
    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Cumulative Return": cumulative - 1,
    }


def plot_results(backtest_df: pd.DataFrame, output_path) -> None:
    df = backtest_df.copy()
    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)

    # Convert cumulative returns to percent
    strat_pct = (df["strategy_cum"] - 1) * 100
    bench_pct = (df["benchmark_cum"] - 1) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["date"], strat_pct, label="Strategy")
    ax.plot(df["date"], bench_pct, label="SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Strategy vs SPY Cumulative Performance")
    ax.legend()
    ax.grid(True)

    # Format x-axis as dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
