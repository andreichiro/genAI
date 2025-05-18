# utils/screening_shared.py
# Diagnostics helpers shared by plots and tables
#   • congestion_tail()  – Φ(t) empirical survival CDF   (mean-field Ū series)
#   • psi_peak_series()  – period-level ψ_eff peaks      (any firm-level df)
#   • summarise_series() – common summary stats helper
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Dict

import numpy as np
import pandas as pd


def congestion_tail(U_bar_series: Sequence[float]) -> pd.Series:
    """
    Return the empirical survival CDF Φ(u) = P[Ū ≥ u] evaluated
    at all unique values of *U_bar_series* (sorted ascending).

    Result is a pd.Series indexed by *u* with values in [0,1].
    """
    if not U_bar_series:
        raise ValueError("U_bar_series must be non-empty")

    u = np.sort(np.asarray(U_bar_series, dtype=float))
    n = u.size
    surv = 1.0 - np.arange(1, n + 1) / n
    return pd.Series(surv, index=u, name="Phi_tail")

def psi_peak_series(df_firms: pd.DataFrame,
                    *, kpi_col: str = "psi_eff") -> pd.Series:
    """
    Group *df_firms* by period *t* and return the **maximum** ψ_eff
    observed across firms in that period.

    Parameters
    ----------
    df_firms : pd.DataFrame
        Long-format frame with at least columns ['t', kpi_col].
    kpi_col : str, default 'psi_eff'
        Column that holds firm-level throughput.

    Returns
    -------
    pd.Series
        Index = t, value = max ψ_eff(t).
    """
    if kpi_col not in df_firms.columns:
        raise KeyError(f"'{kpi_col}' column absent from dataframe")

    return (
        df_firms.groupby("t", sort=True)[kpi_col]
                .max()
                .rename("psi_peak")
    )

def summarise_series(series: Iterable[float]) -> Dict[str, float]:
    """
    Convenience helper used by table_exporter – returns a dict with
    max, mean, and 95-th percentile of any 1-D series.
    """
    arr = np.asarray(list(series), dtype=float)
    if arr.size == 0:
        raise ValueError("series must be non-empty")
    return {
        "max"   : float(np.max(arr)),
        "mean"  : float(np.mean(arr)),
        "p95"   : float(np.percentile(arr, 95)),
    }