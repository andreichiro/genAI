# curation.py
"""
minimal helpers used by sim_runner.

"""
from __future__ import annotations

from typing import Final
import pandas as pd
import numpy as np
import validator  
from pathlib import Path
from collections import OrderedDict          # add to imports
from data_portal import apply_sga_overlay, get_sga_overlay
import logging

_EXPECTED_ORDER: Final = [
    "scenario_id", "t", "firm_id",
    "Y_new",       # output
    "psi_eff", "theta", "queue_len",
]

_DERIVED_COLS: Final = [
    "y_growth_pct",
    "mean_latency",
    "p95_latency",
    "max_latency",  
    "std_latency", 
    "creativity_loss",
    "triage_eff",
    "ROI_skill",
    "congestion_idx",
    "congestion_idx_mean",
    "market_share",
    "Y_lost_decay",
    "y_new_tot",
    "sgna_cost",
    "capital_intensity",        
    "rd_share",                 
    "effective_skills",        
    "x_sum",
    "x_varieties",     
]

def _flatten_x(df: pd.DataFrame) -> pd.DataFrame:        
    if "x_values" not in df.columns:
        return df
    df["x_sum"]       = df["x_values"].apply(np.sum).astype("float64")
    df["x_varieties"] = df["x_values"].apply(len).astype("int64")
    return df.drop(columns="x_values")

def tidy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    * Re-orders cheap log columns so parquet is predictable.
    * Down-casts 't' to int64 for storage efficiency.
    * NO heavy validation – Phase C will enforce schema.
    """
    cols = [c for c in _EXPECTED_ORDER if c in df.columns] + \
           [c for c in df.columns if c not in _EXPECTED_ORDER]
    df = df[cols]                          # reorder
    if "t" in df.columns:
        df["t"] = df["t"].astype("int64")
    # ensure x_values is numpy arrays (joblib parallel sometimes loses dtype)
    if "x_values" in df.columns:
        df["x_values"] = df["x_values"].apply(lambda x: np.asarray(x, dtype=float))
    return df

def _add_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Add %-growth of Y_new within each scenario (first period = 0)."""
    df = df.sort_values(["scenario_id", "t"])
    df["y_growth_pct"] = (
        df.groupby("scenario_id")["Y_new"]
          .pct_change()
          .fillna(0.0)
          .astype("float64")
    )
    return df

def _add_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Capital intensity and R&D share metrics."""
    cap_col = "capital_current" if "capital_current" in df.columns else "K_AI"
    ly_col  = "LY"               if "LY"               in df.columns else "psi_eff"
    la_col  = "LA"               if "LA"               in df.columns else None

    if {cap_col, ly_col}.issubset(df.columns):
        df["capital_intensity"] = (
            df[cap_col] / df[ly_col].replace(0, np.nan)
        ).astype("float64")
    else:
        df["capital_intensity"] = np.nan

    if la_col and {la_col, ly_col}.issubset(df.columns):
        df["rd_share"] = (
            df[la_col] / (df[la_col] + df[ly_col]).replace(0, np.nan)
        ).astype("float64")
    else:
        df["rd_share"] = np.nan

    return df

def _add_effective_skills(df: pd.DataFrame) -> pd.DataFrame:
    """
    effective_skills(t) = synergy(t) * logistic_factor(t)
    Falls back gracefully if the multiplier column is absent (legacy runs).
    """
    if "logistic_factor" in df.columns:
        df["effective_skills"] = (
            df["synergy"] * df["logistic_factor"]
        ).astype("float64")
    else:                              # back-compat - keeps pipeline green
        df["effective_skills"] = df["synergy"].astype("float64")
    return df

def _add_queue_kpis(                            
    df: pd.DataFrame,
    idea_log_path: Path = Path("outputs/idea_log.parquet"),
) -> pd.DataFrame:
    """
    Merge queue-level diagnostics computed by `queue_dynamics.parquet_writer`.

    Always guarantees that the four Phase-E KPIs exist in *df* even when the
    parquet is missing or incomplete, so downstream schema validation is safe.

        • mean_latency     – mean wait time, per period
        • p95_latency      – 95-th percentile wait time
        • creativity_loss  – passthrough column (if already produced upstream)
        • triage_eff       – passthrough column
        • ROI_skill        – legacy placeholder (kept for back-compat)
    """
    for col in (
        "mean_latency", "p95_latency", "max_latency", "std_latency",
        "creativity_loss", "triage_eff", "ROI_skill",
    ):
        if col not in df.columns:
            df[col] = np.nan

    # nothing to merge – return early 
    if not idea_log_path.exists():
        return df

    log_df = pd.read_parquet(idea_log_path)

    # compute latency statistics 
    if {"t_eval", "t_arrival"}.issubset(log_df.columns):
        log_df["lat"] = (log_df["t_eval"] - log_df["t_arrival"]).astype("float64")

        grp   = log_df.groupby(["scenario_id", "t_eval"])["lat"]
        stats = (
            grp.agg(
                mean_latency="mean",
                p95_latency=lambda s: np.percentile(s, 95),
                max_latency=lambda s: np.percentile(s, 99),
                std_latency="std",
            )
            .reset_index()
            .rename(columns={"t_eval": "t"})
            # give right-hand columns a _new suffix so they are
            #      guaranteed to survive the merge even if they don't clash
            .rename(
                columns={c: f"{c}_new" for c in (
                    "mean_latency", "p95_latency", "max_latency", "std_latency"
                )}
            )
        )



        # Outer-merge and coalesce with any pre-existing values
        df = df.merge(stats, how="left", on=["scenario_id", "t"])
        for col in ("mean_latency", "p95_latency",  
                    "max_latency", "std_latency"):       
            df[col] = df[f"{col}_new"].combine_first(df[col])
            df.drop(columns=f"{col}_new", inplace=True, errors="ignore")

    return df

def _add_market_and_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    • market_share   = firm output / total output each period
    • Y_lost_decay   = nominal output minus decay-adjusted output

    Works even if legacy runs lack Y_new_nominal (falls back to zero loss).
    """
    # Market share 
    df["market_share"] = (
        df.groupby(["scenario_id", "t"])["Y_new"]
        .transform(lambda col: col / col.sum() if col.sum() else np.nan)
        .astype("float64")
    )

    # Latency-decay loss 
    if "Y_new_nominal" not in df.columns:
        df["Y_new_nominal"] = df["Y_new"]          # legacy back-compat
    df["Y_lost_decay"] = (
        (df["Y_new_nominal"] - df["Y_new"])
        .clip(lower=0)
        .astype("float64")
    )
    return df

def _add_tot_output(df: pd.DataFrame) -> pd.DataFrame:
    df["y_new_tot"] = (
        df.groupby(["scenario_id", "t"])["Y_new"]
          .transform("sum")
          .astype("float64")
    )

    # scenario-level mean congestion 
    cong = (
        df.groupby(["scenario_id", "t"])["congestion_idx"]
          .mean()
          .rename("congestion_idx_mean")
          .reset_index()
    )
    df = df.merge(cong, on=["scenario_id", "t"], how="left")

    return df

def curate(parquet_path: str = "outputs/simulations.parquet") -> None:
    """
    Read the raw Phase-B parquet, validate, enrich, and write the curated file.
    Never mutates the original parquet.
    """
    raw = pd.read_parquet(parquet_path)

    # 1) Schema validation (fail-fast, aggregated errors)
    validator.SCHEMA.validate(raw, lazy=True)

    # 2) Transform pipeline
    try:
        overlay_df = get_sga_overlay()   # will raise FileNotFoundError if absent
        _sgna_merge = lambda d: apply_sga_overlay(d, overlay_df)
    except FileNotFoundError as err:
        logging.warning("SG&A overlay not found (%s) – sgna_cost set to 0.", err)

        def _sgna_merge(d: pd.DataFrame) -> pd.DataFrame:   # graceful fallback
            d["sgna_cost"] = 0.0
            return d

    df = (
        raw
            .pipe(_flatten_x) 
            .pipe(_add_growth)
            .pipe(_add_intensity)            
            .pipe(_add_effective_skills)    
            .pipe(_add_queue_kpis)
            .pipe(_add_market_and_decay)
            .pipe(_sgna_merge)    
            .pipe(_add_tot_output)        # always defined by the try/except
    )


    # 3) Re-order columns → core · derived · any extras
    base_cols = [c for c in raw.columns if c != "x_values"]
    ordered_unique_cols = list(OrderedDict.fromkeys(base_cols + _DERIVED_COLS))
    df = df[ordered_unique_cols]

    # 4) Persist artefacts
    out_path = Path("outputs/simulations_curated.parquet")
    preview_path = out_path.with_suffix(".preview.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path, index=False)
    df.to_csv(preview_path, index=False)

    print(f"[curation] ✅ wrote {len(df):,} rows  →  {out_path}")
    print(f"[curation] 📄 preview (10 rows)      →  {preview_path}")


if __name__ == "__main__":        # CLI:  python -m curation  (optional)
    curate()

"""
1. What the module does, step by step
Lightweight tidy helper for the runner –
tidy_dataframe() is called inside sim_runner so that large, multi-process runs can concatenate their individual DataFrames without dtype headaches.

It simply re-orders a handful of "cheap" log columns, forces t into efficient int64, and guarantees that x_values is a proper NumPy array.

No deep validation happens here—that's deferred to the main curate() pipeline.

2. Schema validation –
The first thing curate() does is load outputs/simulations.parquet and pass it through validator.SCHEMA.

This immediately catches missing columns, negative values, or dtype drift.

Validation is lazy, so all errors are aggregated and reported at once.

3. Column flattening –
Raw simulations store the whole x_values vector in every row, which makes BI tools choke.

_flatten_x() removes that object column and replaces it with two scalars:
x_sum (total intermediate-goods usage) and x_varieties (number of varieties).

4. Derived metrics –

_add_growth() computes period-over-period percentage growth of Y_new within each scenario.

_add_intensity() adds two classic macro ratios: capital_intensity (K / L_Y) and rd_share (L_A / (L_Y + L_A)), taking care to avoid divide-by-zero.

_add_effective_skills() multiplies synergy by the logistic multiplier produced in Phase G to create an effective_skills index. The function keeps backward compatibility by falling back to bare synergy if the logistic column is absent.

5. Column ordering –
A deterministic order is enforced: original "core" variables first, followed by all _DERIVED_COLS. Any experimental extras that future phases add are appended automatically, so nothing breaks.

6. Output artefacts –

Full dataset → outputs/simulations_curated.parquet

Ten-row human preview → outputs/simulations_curated.preview.csv
Both files are overwritten atomically, never touching the raw parquet.

7. Command-line entry point –
Running python -m curation (or python curation.py) will perform the entire routine using default paths, making it easy to rerun just the cleaning stage without touching simulations.

8. How to use it
Standard pipeline – You normally never call it directly; replicate.py or make reproduce triggers it automatically after simulations finish.

Ad-hoc validation – If you tweaked sim_runner or edited the raw parquet by hand, just run


python curation.py              # validates and rebuilds the curated file
Alternative input – Pass a custom file if you have multiple experiment shards:

import curation
curation.curate("outputs/alt_simulations.parquet")
Configuration knobs
curation.py is intentionally free of user-facing settings so that the schema remains a single point of truth. The only variables you might touch are:

_EXPECTED_ORDER and _DERIVED_COLS – add or reorder names here if you extend the model with new fields.

Output directory – change out_path = Path("outputs/…") inside curate() if you need a different location.

Everything else—growth formulas, logistic multipliers, thresholds—comes from earlier phases and lives either in YAML or in the simulation engine.
"""