import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def select_high_rsi_cluster(df: pd.DataFrame, rsi_init: list[int]) -> pd.DataFrame:
    df = df.copy()
    selected = df.dropna(subset=["RSI", "bollinger_band_width", "ATR_zscore", "MACD_zscore", "beta_mkt"]).copy()
    feature_columns = ["RSI", "bollinger_band_width", "ATR_zscore", "MACD_zscore", "beta_mkt"]
    scaler = StandardScaler()
    X = scaler.fit_transform(selected[feature_columns])
    anchor_rsi = np.array(rsi_init, dtype=float)
    rsi_scaled = (anchor_rsi - scaler.mean_[0]) / scaler.scale_[0]
    init_centers = np.zeros((len(anchor_rsi), X.shape[1]))
    init_centers[:, 0] = rsi_scaled
    kmeans = KMeans(
        n_clusters=len(anchor_rsi),
        init=init_centers,
        n_init=1,
        random_state=42,
        max_iter=300,
    )
    selected["cluster"] = kmeans.fit_predict(X)
    selected["cluster_center_rsi"] = kmeans.cluster_centers_[:, 0][selected["cluster"]]
    # cluster centers (RSI component) after convergence
    centers_rsi = kmeans.cluster_centers_[:, 0]
    print("[Clustering] cluster RSI centroids:")
    for idx, val in enumerate(centers_rsi):
        print(f"  cluster {idx}: RSI centroid (scaled) = {val:.6f}")

    best_cluster = int(np.argmax(centers_rsi))
    # Debug: show per-cluster counts per month (before filtering to best)
    try:
        all_labels = pd.DataFrame({"date": selected["date"], "ticker": selected["ticker"], "cluster": selected["cluster"]})
        counts_by_month_cluster = all_labels.groupby(["date", "cluster"]) ["ticker"].nunique().unstack(fill_value=0).sort_index()
        print("[Clustering] counts by month and cluster (sample):")
        # print first 10 months to avoid huge output
        for d, row in counts_by_month_cluster.head(10).iterrows():
            rowstr = ", ".join([f"c{c}:{int(v)}" for c, v in row.items()])
            print(f"  {d}: {rowstr}")
    except Exception as e:
        print(f"[Clustering] per-month cluster counts debug failed: {e}")

    # If a specific cluster (e.g., 3) is empty, print a warning
    try:
        if len(centers_rsi) > 3:
            members = (selected["cluster"] == 3).sum()
            if members == 0:
                print("[Clustering] WARNING: cluster 3 has zero members after fit")
            else:
                print(f"[Clustering] cluster 3 members: {members}")
    except Exception:
        pass

    selected = selected[selected["cluster"] == best_cluster].copy()
    selected["selected_universe"] = True

    # Debug: print diagnostic info about selection
    try:
        print(f"[Clustering] total rows with features: {len(selected)}")
        print(f"[Clustering] unique months in selection: {selected['date'].nunique()}")
        print("[Clustering] sample selection rows:")
        print(selected[ ["date","ticker","RSI","cluster"] ].head(10).to_string(index=False))
        counts = selected.groupby("date")["ticker"].nunique().sort_index()
        print("[Clustering] Selected counts (best cluster) per month:")
        if counts.empty:
            print("  (none)")
        else:
            for d, c in counts.items():
                print(f"  {d}: {c}")
    except Exception as e:
        print(f"[Clustering] debug failed: {e}")
    df = df.merge(
        selected[["ticker", "date", "selected_universe"]],
        on=["ticker", "date"],
        how="left",
    )
    df["selected_universe"] = df["selected_universe"].fillna(False)
    return df
