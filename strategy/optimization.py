import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


def optimize_portfolio(price_df: pd.DataFrame) -> pd.Series:
    price_df = price_df.copy().dropna(axis=1, how="any")
    # require at least one column; if only one, allocate fully to it
    if price_df.shape[1] == 0:
        return pd.Series(dtype=float)
    if price_df.shape[1] == 1:
        return pd.Series(1.0, index=price_df.columns)

    mu = expected_returns.mean_historical_return(price_df, frequency=12)
    S = risk_models.sample_cov(price_df, frequency=12)
    n = len(mu)
    # set sensible bounds that won't be infeasible for small n
    min_weight = 0.0
    max_weight = 0.5
    try:
        print(f"[Optimize] price_df shape={price_df.shape}, columns={list(price_df.columns)}")
        print(f"[Optimize] mu length={len(mu)}, mu sample=\n{mu.head().to_string()}" if len(mu)>0 else "[Optimize] mu empty")
        print(f"[Optimize] cov shape={S.shape}")
        ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
        weights = ef.max_sharpe(risk_free_rate=0.0)
        cleaned = ef.clean_weights()
        print(f"[Optimize] cleaned weights (raw dict): {cleaned}")
        if not cleaned:
            # fallback to equal weights if optimizer returned no cleaned weights
            print("[Optimize] cleaned weights empty — falling back to equal weights")
            return pd.Series(1.0 / n, index=price_df.columns).reindex(price_df.columns).fillna(0.0)
        return pd.Series(cleaned).reindex(price_df.columns).fillna(0.0)
    except Exception:
        weights = pd.Series(1.0 / n, index=price_df.columns)
        print("[Optimize] exception during optimization — using equal weights")
        return weights
