# data_portal.py
# External-data plug-in for the ECB refactor
# Purpose
# • Provide a *single* access-point for any empirical overlays
#   (SG&A / OPEX, macro price indices, energy costs …).
# • Keep all file-system paths in one place and honour the project’s
#   YAML-driven   paths:  block for CI portability.
#

# Implements SG&A overlays:
#   • A CSV file with columns  scenario_id, t, sga_pct  (share of revenue)
#   • An accessor `get_sga_overlay()` that loads & validates the file.
#   • A helper `apply_sga_overlay()` to merge the overlay onto a
#     simulations DataFrame and add a column  sgna_cost.
#

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Callable

import pandas as pd
import pandera as pa
from pandera import Column, Check
import yaml

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve all external-data directories from the canonical scenarios.yaml
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_CFG_YAML = _ROOT / "scenarios.yaml"

_DEFAULT_DIR = _ROOT / "externals"
_DEFAULT_DIR.mkdir(exist_ok=True, parents=True)          # ensure idempotent

try:
    _CFG = yaml.safe_load(_CFG_YAML.read_text()) if _CFG_YAML.exists() else {}
    _PATHS = (_CFG or {}).get("paths", {}) if isinstance(_CFG, dict) else {}
except Exception as _e:                                   # pragma: no cover
    logging.warning("Could not parse %s (%s) – falling back to defaults",
                    _CFG_YAML, _e)
    _PATHS = {}

_EXT_DIR = Path(_PATHS.get("extern_data", _DEFAULT_DIR)).resolve()

# ---------------------------------------------------------------------------
# SG&A overlay loader  (Phase 8 deliverable)
# ---------------------------------------------------------------------------
_SGA_SCHEMA = pa.DataFrameSchema(
    {
        "scenario_id": Column(str, nullable=False),
        "t":            Column(int, Check.ge(0), nullable=False),
        "sga_pct":      Column(float, Check.ge(0), Check.le(1), nullable=False),
    },
    strict=True,
    coerce=True,
    name="SGAOverlay",
)


def _load_sga_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Low-level CSV reader + validator.

    Parameters
    ----------
    csv_path : pathlib.Path, optional
        Custom location; defaults to  *_EXT_DIR/sga_overlays.csv*.

    Returns
    -------
    pandas.DataFrame  – validated against `_SGA_SCHEMA`
    """
    path = csv_path or (_EXT_DIR / "sga_overlays.csv")
    if not path.exists():
        raise FileNotFoundError(f"SG&A overlay file not found: {path}")

    df = pd.read_csv(path)
    _SGA_SCHEMA.validate(df, lazy=True)
    return df


def get_sga_overlay(scenario_id: str | None = None,
                    *,
                    csv_path: Path | None = None) -> pd.DataFrame:
    """
    Public accessor used by downstream simulation curation.

    Parameters
    ----------
    scenario_id : str, optional
        If given, returns only the rows for that scenario.
    csv_path : Path, optional
        Override default CSV location.

    Returns
    -------
    pandas.DataFrame
        Always contains columns  ["scenario_id", "t", "sga_pct"].
    """
    df = _load_sga_csv(csv_path)
    if scenario_id is not None:
        df = df.loc[df["scenario_id"] == scenario_id].reset_index(drop=True)
    return df


def apply_sga_overlay(sim_df: pd.DataFrame,
                      overlay_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SG&A overlay onto *sim_df* and compute **sgna_cost**:

        sgna_costₜ = sga_pctₜ  ×  Y_newₜ

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Output from curation.py (must include Y_new, t, scenario_id).
    overlay_df : pandas.DataFrame
        Output from `get_sga_overlay()` with matching keys.

    Returns
    -------
    pandas.DataFrame
        Original columns plus *sgna_cost*.  The function is *pure*:
        it never mutates its arguments.
    """
    required_cols = {"scenario_id", "t", "Y_new"}
    missing = required_cols - set(sim_df.columns)
    if missing:
        raise KeyError(f"sim_df missing columns: {sorted(missing)}")

    merged = pd.merge(sim_df, overlay_df, on=["scenario_id", "t"],
                      how="left", validate="m:m")
    if merged["sga_pct"].isna().any():
        _LOG.warning("SG&A overlay has gaps – missing rows filled with 0.")
        merged["sga_pct"] = merged["sga_pct"].fillna(0.0)

    merged["sgna_cost"] = merged["sga_pct"] * merged["Y_new"]
    return merged


# ---------------------------------------------------------------------------
# Registry for future external datasets
# ---------------------------------------------------------------------------
_LOADER_REGISTRY: Dict[str, Callable[..., pd.DataFrame]] = {
    "sga": get_sga_overlay,
    # future keys: "energy", "wacc", …
}


def get_external(name: str, **kwargs) -> pd.DataFrame:
    """
    Generic factory so caller code can request *any* external dataset
    without importing private helpers.

        df_energy = data_portal.get_external("energy", scenario_id="baseline")

    Raises `KeyError` if the dataset is unknown.
    """
    if name not in _LOADER_REGISTRY:
        raise KeyError(f"External dataset '{name}' not registered")
    return _LOADER_REGISTRY[name](**kwargs)  # type: ignore[arg-type]
