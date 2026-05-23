from pathlib import Path

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
TOP_N_LIQUID = 150
HORIZONS = [1, 2, 3, 6, 9, 12]
BETAS_WINDOW = 24
KMEANS_CLUSTERS = 4
KMEANS_RSI_INIT = [30, 45, 55, 70]
MIN_WEIGHT_FACTOR = 0.5
MAX_WEIGHT = 0.10
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_PATH = RESULTS_DIR / "strategy_performance.png"
BENCHMARK_TICKER = "SPY"
