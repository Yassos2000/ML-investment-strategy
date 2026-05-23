import io
import re
import zipfile

import pandas as pd
import requests
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm


def download_fama_french_factors(start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    filename = next((n for n in archive.namelist() if n.endswith(".csv")), None)
    if filename is None:
        raise ValueError("Could not find CSV file inside Fama-French archive")
    raw = archive.open(filename).read().decode("latin1").splitlines()
    header_pattern = re.compile(r"^(?:Date|Period|,).*\bMkt-RF\b.*\bSMB\b.*\bHML\b.*\bRF\b", re.I)
    header_idx = next((i for i, row in enumerate(raw) if header_pattern.search(row)), None)
    if header_idx is None:
        raise ValueError("Could not locate the header row in the Fama-French CSV")
    data = "\n".join(raw[header_idx:])
    factors = pd.read_csv(io.StringIO(data), index_col=0, skipinitialspace=True)
    index_strings = factors.index.astype(str).str.strip()
    valid_monthly = index_strings.str.match(r"^\d{6}$")
    factors = factors.loc[valid_monthly]
    factors.index = pd.to_datetime(
        factors.index.astype(str).str.strip(),
        format="%Y%m",
        errors="raise",
    ).to_period("M").to_timestamp("M", how="end")
    factors = factors.apply(pd.to_numeric, errors="coerce")
    factors.index.name = "date"
    return factors.loc[start_date:end_date]


def calculate_rolling_betas(df: pd.DataFrame, factors: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M", how="end")
    df = df.sort_values(["ticker", "date"]).copy()
    factors = factors.copy()
    factors.index = pd.to_datetime(factors.index)
    betas = []
    for ticker, group in df.groupby("ticker", sort=False):
        group = group.copy()
        group = group.merge(
            factors,
            left_on="date",
            right_index=True,
            how="left",
        )
        group["monthly_return"] = group["adjClose"].pct_change(1)
        group = group.dropna(subset=["monthly_return", "Mkt-RF", "SMB", "HML", "RMW", "CMA"])
        if len(group) < window:
            betas.append(group)
            continue
        exog = sm.add_constant(group[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
        endog = group["monthly_return"] - group["RF"] / 100.0
        model = RollingOLS(endog, exog, window=window)
        results = model.fit()
        params = results.params
        params = params.reset_index(drop=True)
        params.index = group.index[-len(params) :]
        params = params.rename(columns={"Mkt-RF": "beta_mkt", "SMB": "beta_smb", "HML": "beta_hml", "RMW": "beta_rmw", "CMA": "beta_cma"})
        group = group.join(params)
        betas.append(group)
    return pd.concat(betas).reset_index(drop=True)
