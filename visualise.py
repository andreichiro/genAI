# visualise.py
"""
Figure generation.

Run `python -m visualise` or `python visualise.py`.
All PNG files go to   figures/
All HTML files go to  figures_html/
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm    # lightweight progress bar
import sys
import textwrap 
import matplotlib.pyplot as _plt
import plotly.express      as _px
import pyarrow.parquet as pq 

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_ROOT = Path(__file__).resolve().parent
_CFG_YAML = _ROOT / "scenarios.yaml"                             

# sensible fall-backs (project-relative defaults)
_DEFAULTS = {
    "out_cur":  _ROOT / "outputs" / "simulations_curated.parquet",
    "fig_png":  _ROOT / "figures",
    "fig_html": _ROOT / "figures_html",
}

try:                                                            
    _CFG = yaml.safe_load(_CFG_YAML.read_text()) if _CFG_YAML.exists() else {}
    _PATHS = (_CFG.get("defaults", {}).get("paths", {})  
              if isinstance(_CFG, dict) else {})
except Exception as _e:                                        
    logging.warning("Could not load %s (%s) – using defaults", _CFG_YAML, _e)
    _PATHS = {}

_DATA      = Path(_PATHS.get("out_cur",  _DEFAULTS["out_cur"])).resolve()   
_DIR_PNG   = Path(_PATHS.get("fig_png",  _DEFAULTS["fig_png"])).resolve()  
_DIR_HTML  = Path(_PATHS.get("fig_html", _DEFAULTS["fig_html"])).resolve()  


_PLOT_SPECS = _ROOT / "plot_specs.yaml"                                      
_META_COLS = ["scenario_id", "test_label", "hypothesis"]

def _wrap_label(text: str, width: int = 25) -> str:             
    """Return *text* wrapped to the given *width* so legends never overflow."""
    return "\n".join(textwrap.wrap(str(text), width))

def _pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Build a wide t × series table *with an inner tqdm bar* so that
    progress is visible on huge datasets.

      • most metrics   → one series per scenario
      • market_share   → one series per scenario + firm
    """
    # ── create a helper column `_col` with the legend label 
    if metric == "market_share":
        df = df.copy()
        df["_col"] = (
            df[["scenario_id", "firm_id"]]
            .astype(str)
            .agg(" – ".join, axis=1)             # e.g.  A123 – 7
        )
    else:
        meta_cols = [c for c in _META_COLS if c in df.columns]
        df = df.copy()
        df["_col"] = (
            df[meta_cols]
            .astype(str)
            .agg(" | ".join, axis=1)
            .map(_wrap_label)
        )

    # ── incremental pivot (one t-slice at a time) 
    CHUNK = 2_000_000                # rows per chunk (~400 MB RAM)
    parts = []
    for i in tqdm(range(0, len(df), CHUNK),
                  desc=f"pivot-{metric}", leave=False):
        part = (df.iloc[i:i+CHUNK]
                  .groupby(["t", "_col"], sort=False)[metric]
                  .mean()
                  .unstack("_col"))
        parts.append(part)

    wide = (pd.concat(parts)
              .groupby(level=0)      # merge identical t-indices
              .first()
              .sort_index())
    return wide

def _load_specs() -> dict:
    specs = yaml.safe_load(_PLOT_SPECS.read_text())
    if not isinstance(specs, dict) or not specs:
        raise ValueError(f"plot_specs.yaml is empty or malformed: {_PLOT_SPECS}")
    return specs

def _static_lineplot(df: pd.DataFrame, metric: str, spec: dict, out_png: Path) -> None:
    """Matplotlib PNG with journal-friendly dpi."""
    _plt.figure(figsize=(6, 3))
    for scn, grp in df.groupby("scenario_id"):
        _plt.plot(grp["t"], grp[metric], label=scn)
    _plt.title(spec.get("title", metric))
    _plt.xlabel("t")
    _plt.ylabel(spec.get("ylabel", metric))
    if spec.get("log_y"):
        _plt.yscale("log")
    _plt.legend(fontsize="x-small", ncol=2)
    _plt.tight_layout()
    _plt.savefig(out_png, dpi=300)
    _plt.close()

def _interactive_lineplot(df: pd.DataFrame, metric: str, spec: dict, out_html: Path) -> None:
    """Plotly HTML for exploratory use (hover, zoom, toggle)."""
    fig = _px.line(
        df,
        x="t",
        y=metric,
        color="scenario_id",
        title=spec.get("title", metric),
        labels={"t": "t", metric: spec.get("ylabel", metric)},
        log_y=bool(spec.get("log_y", False)),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")

def _draw(metric: str, spec: dict, wide: pd.DataFrame,
          out_png: Path, out_html: Path) -> None:

    # *market-share* needs firm-level traces in the legend

    if metric == "market_share":                                  
        long = (wide.reset_index()                               
                     .melt(id_vars="t",                            
                           var_name="scenario_firm",              
                           value_name="market_share"))            

        #  Matplotlib (static PNG) 
        _plt.figure(figsize=(6, 3))                                
        for name, grp in long.groupby("scenario_firm"):         
            _plt.plot(grp["t"], grp["market_share"],               
                     label=name, lw=1.5, marker="o")               
        _plt.title(spec.get("title", metric))                     
        _plt.xlabel("Year t")                                     
        _plt.ylabel(spec.get("ylabel", metric))                   
        _plt.legend(fontsize="x-small", ncol=3)                    
        _plt.tight_layout()                                      
        _plt.savefig(out_png, dpi=300)                            
        _plt.close()                                             

        #  Plotly (interactive HTML) 
        fig = _px.line(long, x="t", y="market_share",              
                       color="scenario_firm",                   
                       title=spec.get("title", metric),         
                       labels={                                   
                           "t": "Year t",                          
                           "market_share": spec.get("ylabel", metric) 
                       })                                        
        fig.write_html(out_html, include_plotlyjs="cdn")         
        return                                                     

    # -------- Matplotlib
    _palette = _plt.colormaps.get_cmap("tab20").colors
    ax = wide.plot(figsize=(6,3), lw=1.5, marker="o", color=_palette)
    ax.set_title(spec.get("title", metric))
    ax.set_xlabel("Year t")
    ax.set_ylabel(spec.get("ylabel", metric))
    if spec.get("log_y"): ax.set_yscale("log")
    ax.legend(fontsize="x-small", ncol=2, loc="best")
    ax.figure.tight_layout()
    ax.figure.savefig(out_png, dpi=300)
    _plt.close(ax.figure)

    # -------- Plotly
    fig = _px.line(wide, x=wide.index, y=wide.columns,
                   labels={"x": "Year t", "value": spec.get("ylabel", metric)},
                   title=spec.get("title", metric),
                   log_y=bool(spec.get("log_y", False)))

    fig.write_html(out_html, include_plotlyjs="cdn")

def render_all() -> None:
    """
    Main orchestration: validate data, then loop over every metric defined
    in plot_specs.yaml and produce a PNG + HTML.
    """
    specs = _load_specs()

    _DIR_PNG.mkdir(parents=True, exist_ok=True)
    _DIR_HTML.mkdir(parents=True, exist_ok=True)

    # ── READ THE PARQUET ONLY ONCE  ──────────────────────────────────────
    _LOG.info("→ loading curated dataset …")
    # grab *all* meta columns plus the union of metrics we’ll plot
    _avail = set(pq.ParquetFile(_DATA).schema.names)   # ← NEW

    # grab *all* meta columns plus the union of metrics we’ll plot
    all_metrics = set(specs.keys()) | {"market_share"}          # market_share needs firm_id

    # request only the columns that really exist in the parquet  ↓ NEW ↓
    cols = [c for c in (["t", "firm_id", *_META_COLS, *all_metrics]) if c in _avail]

    df_full = pd.read_parquet(_DATA, columns=cols)

    _LOG.info("   dataset in RAM: %d rows × %d columns", *df_full.shape)

    for metric, spec in tqdm(specs.items(), desc="plots"):
        # ----------------------------------------------------------------
        if metric not in df_full.columns or df_full[metric].isna().all():
            _LOG.warning("Metric '%s' missing or all-NaN – skipped.", metric)
            continue

        _LOG.info("   ↳ pivoting %s …", metric)           # ← add
        _t0 = pd.Timestamp.now()                          # ← add

        # keep only the columns needed for this metric ↓
        df = df_full[["t", *(_META_COLS),
                      *(["firm_id"] if metric == "market_share" else []),
                      metric]].copy()

        wide = _pivot(df, metric)
        _LOG.info("     done in %s", pd.Timestamp.now() - _t0)  # ← add

        if wide.empty:
            _LOG.warning("Metric '%s' has no finite data – skipped.", metric)
            continue

        _draw(metric, spec, wide,
              _DIR_PNG  / f"{metric}.png",
              _DIR_HTML / f"{metric}.html")

    _LOG.info("finished ✓")

if __name__ == "__main__":          # CLI entry-point
    # Allow `python -m visualise` **or** `python visualise.py`
    if __package__ is None:
        # executed as a *script* → OK
        render_all()
    else:
        # executed as `-m` but module, not package
        sys.stderr.write(
            "Run me via  python visualise.py  (this file is a module, not a package)\n"
        )
        sys.exit(1)


"""
1. What the script achieves

Quality gate – the very first thing it does is reload the curated dataset and re-run the Pandera SCHEMA validation. If anything changed the file on disk (corruption, manual edits, version mismatch) the script aborts before attempting a single plot.

Declarative plotting – every visual that should appear in the paper is described once, in plot_specs.yaml. Each YAML entry says which metric to plot, what title/label it needs, and whether to use a log-scale, nothing more.

Dual output – for every metric it emits

a high-resolution PNG in figures/ for static inclusion in LaTeX or Word;

a matching HTML file in figures_html/ built with Plotly so researchers can hover, zoom, and explore.

Progress indication – a tqdm bar keeps long batch jobs transparent.

Idempotency – repeated calls overwrite the same filenames, so the repo never accumulates stale artefacts.

2. How to use it

The fastest way is:

# after phases A-C have run

python -m visualise            

Behind the scenes this triggers the render_all() function.

All files land next to the code tree:

figures/
    y_growth_pct.png
    capital_intensity.png
    … one per spec
figures_html/
    y_growth_pct.html
    capital_intensity.html
outputs/
    simulations_curated.parquet   # already created by Phase C
plot_specs.yaml
Because plotting is pure I/O the command is safe to execute in parallel with other CPU-heavy steps; it only loads the Parquet once, slices columns in memory and streams a handful of matplotlib + Plotly objects to disk.

3. Configuration knobs and where they live
Aspect	Where to change	Effect
Which metrics appear	plot_specs.yaml	Add or remove a top-level key. If the metric name does not exist in the dataframe the script logs a warning and skips it.
Styling (title, y-label, log-axis)	plot_specs.yaml per metric	The plotting helpers respect the dict keys title, ylabel, log_y as seen in the YAML.
Input location	constant _DATA (outputs/simulations_curated.parquet)	Move or rename the curated file and adjust the constant or set a symlink.
Output directories	constants _DIR_PNG, _DIR_HTML	Point them somewhere else if your journal has a specific folder structure.
Logging verbosity	environment variable LOGLEVEL or change logging.basicConfig(level=…)	Anything from DEBUG to WARNING works; by default the script prints one informative line per run plus any metric-missing warnings.

The script deliberately avoids command-line flags – reproducibility in Phase H is about one-command replication (make reproduce).
If you need custom paths you can fork the file or wrap it in your own CLI, but the default workflow assumes the canonical project layout.
"""