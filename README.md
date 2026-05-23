# ML Investment Strategy

ML Investment Strategy is a complete Python pipeline for backtesting a monthly stock selection strategy on the S&P 500.

## Full Pipeline

1. Download S&P 500 OHLCV data with `yfinance` and cache the results locally.
2. Compute technical indicators including RSI, Bollinger Bands, ATR, MACD, and volatility measures.
3. Aggregate daily data to monthly frequency and select a liquid universe of stocks.
4. Compute multi-horizon returns, clip outliers, and enrich the dataset with Fama-French 5-factor exposures.
5. Estimate rolling factor betas and use asset clustering to pick a high-RSI cluster each month.
6. Optimize monthly portfolio weights using a max Sharpe ratio objective and bound constraints.
7. Backtest the strategy monthly and compare performance against the SPY benchmark.
8. Save performance metrics and a results chart to the `results/` folder.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Strategy

Run the backtest with:

```bash
python main.py
```

The script downloads data, runs the strategy, and writes charts and metrics to `results/`.

## Results

The strategy is evaluated against SPY over the 2018-2024 period. The output includes:

- strategy cumulative return vs SPY
- annualized return and volatility
- Sharpe ratio
- max drawdown

The final performance chart is saved in `results/strategy_performance.png`.
