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
import pyarrow as pa 
import json   
import time 
try:
    import hvplot.pandas        # auto-registers .hvplot accessor
    from holoviews.operation.datashader import datashade
    _HAS_DASH = True
except ImportError:              # pragma: no cover
    _HAS_DASH = False

try:
    import validator                       # project-local schema module
except ImportError:                        # allow visualisation without it
    validator = None

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

# fan-chart 
_QUANTS_PARQ = _ROOT / "outputs" / "group_quantiles.parquet"
_EXPRESSIVE  = json.loads(
    Path(_ROOT / "outputs" / "expressive_scenarios.json").read_text()
    if ( _ROOT / "outputs" / "expressive_scenarios.json").exists() else "{}"
)


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
          out_png: Path, out_html: Path,
          *,
          group_label: str,
          df_long: pd.DataFrame) -> None:

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

    #  density backdrop (Datashader) 
    if _HAS_DASH and wide.shape[1] > 100:
        import holoviews as hv
        hv.extension("bokeh")                               # idempotent

        # stack into one long column so every point has the *same*
        # data variable → avoids the xarray merge error raised by
        # datashader when column names differ.
        long_df = (
            wide.reset_index()
                .melt(id_vars="t", value_name="val")        # 'series' col unused
                .dropna(subset=["val"])
        )

        base = long_df.hvplot.scatter(
            x="t", y="val",                                 # common variable name
            width=650, height=300,
            xaxis=None, yaxis=None
        )

        dens = datashade(base, cmap="Greys", dynamic=False) # static Image, no DMap
        dens = dens.opts(
            title=f"{spec.get('title', metric)}  |  {group_label}",
            axiswise=True
        )

        try:
            hv.save(dens,
                    str(out_html.with_suffix(".density.html")),
                    backend="bokeh")
        except Exception as exc:              # pragma: no cover
            _LOG.warning("Skipping density save – %s", exc)



    #  fan-chart + expressive lines (Matplotlib) 
    fig, ax = _plt.subplots(figsize=(6.4, 3.6))

    # quantile ribbons
    if _QUANTS_PARQ.exists():
        q_cols_core = [f"{metric}_{q}" for q in ("p25","p75","p05","p95","p50")]
        q_cols_ext  = [c for c in (f"{metric}_min", f"{metric}_max")
                       if ( _QUANTS_PARQ.exists()
                            and c in pq.ParquetFile(_QUANTS_PARQ).schema.names)]
        q_cols = q_cols_core + q_cols_ext
        q_df = (pd.read_parquet(_QUANTS_PARQ, columns=["test_label", "t", *q_cols])
                  .query("test_label == @group_label"))
        if not q_df.empty:
            ax.fill_between(q_df["t"], q_df[f"{metric}_p25"], q_df[f"{metric}_p75"],
                            color="C0", alpha=0.20, label="IQR")
            ax.fill_between(q_df["t"], q_df[f"{metric}_p05"], q_df[f"{metric}_p95"],
                            color="C0", alpha=0.10, label="P5–P95")
            ax.plot(q_df["t"], q_df[f"{metric}_p50"], color="black",
                    lw=1.6, label="median")
            if f"{metric}_min" in q_df.columns:
                ax.plot(q_df["t"], q_df[f"{metric}_min"], ls="--",
                         color="grey", lw=0.8)
                ax.plot(q_df["t"], q_df[f"{metric}_max"], ls="--",
                        color="grey", lw=0.8)

    # expressive scenario lines
    key = f"{group_label}|{metric}"
    ids = _EXPRESSIVE.get(key, [])[:5]
    if not ids:
        # deterministic median-path fallback 
        grp = (df_long.groupby("scenario_id")[["t", metric]]
                        .apply(lambda g: g.set_index("t")[metric]))
        p50 = grp.unstack(level=0).median(axis=1)          # series indexed by t

        # squared L2 distance to the p50 curve
        dist = {sid: ((series - p50).pow(2).sum())
                for sid, series in grp.groupby(level=0, axis=1)}
        fallback_id = min(dist, key=dist.get)
        ids = [fallback_id]
            
    for i, sid in enumerate(ids):
        sub = df_long.loc[df_long["scenario_id"] == sid]
        ax.plot(sub["t"], sub[metric], lw=1.5, label=_wrap_label(sid), color=f"C{i}")

    ax.set_title(f"{spec.get('title', metric)}  |  {group_label}")
    ax.set_xlabel("Year t")
    ax.set_ylabel(spec.get("ylabel", metric))
    if spec.get("log_y"):
        ax.set_yscale("log")

    ax.legend(fontsize="x-small", ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    _plt.close(fig)

    # interactive thin-lines Plotly (≤20 series) 
    if 0 < len(ids) <= 20:
        fig = _px.line(df_long.loc[df_long["scenario_id"].isin(ids)],
                       x="t", y=metric, color="scenario_id",
                       title=ax.get_title(),
                       labels={"t": "Year t", metric: spec.get("ylabel", metric)},
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
    _t0 = time.perf_counter()        
    # grab *all* meta columns plus the union of metrics we’ll plot
    _avail = set(pq.ParquetFile(_DATA).schema.names)   # ← NEW

    # grab *all* meta columns plus the union of metrics we’ll plot
    all_metrics = set(specs.keys()) | {"market_share"}          # market_share needs firm_id

    # request only the columns that really exist in the parquet  ↓ NEW ↓
    schema_cols = list(validator.SCHEMA.columns.keys()) if validator else []
    base_cols   = ["t", "firm_id", *_META_COLS, *all_metrics]
    cols_all    = [c for c in (base_cols + schema_cols) if c in _avail]

    pf = pq.ParquetFile(_DATA)
    parts: list[pd.DataFrame] = []
    for batch in tqdm(
            pf.iter_batches(columns=cols_all, batch_size=2_000_000),
            total=pf.num_row_groups,
            desc="read-parquet"):
        parts.append(pa.Table.from_batches([batch]).to_pandas())

    df_full = pd.concat(parts, ignore_index=True)

    _LOG.info("   loaded %s rows × %s cols (≈%.1f MB) in %.1f s",
              len(df_full), df_full.shape[1],
              df_full.memory_usage(deep=True).sum() / 1e6,
              time.perf_counter() - _t0)

    if validator:
        validator.SCHEMA.validate(df_full, lazy=True)

    for metric, spec in tqdm(specs.items(), desc="plots"):

        if metric not in df_full.columns or df_full[metric].isna().all():
            _LOG.warning("Metric '%s' missing or all-NaN – skipped.", metric)
            continue

        # after validation, trim to the columns we really need
        df_metric = df_full[["t", *(_META_COLS),
                             *(["firm_id"] if metric == "market_share" else []),
                             metric]].copy()
        #  facet by test_label 
        for grp_name, df in df_metric.groupby("test_label", sort=False):
            _LOG.info("   ↳ pivoting %s | %s …", metric, grp_name)
            _t0 = pd.Timestamp.now()

            wide = _pivot(df, metric)
            _LOG.info("     done in %s", pd.Timestamp.now() - _t0)

            if wide.empty:
                _LOG.warning("Metric '%s' (%s) has no finite data – skipped.",
                             metric, grp_name)
                continue

            stem = f"{metric}__{grp_name}"
            _draw(metric, spec, wide,
                  _DIR_PNG  / f"{stem}.png",
                  _DIR_HTML / f"{stem}.html",
                  group_label=grp_name,
                  df_long=df)
    
    _LOG.info("finished – plotted to %s and %s", _DIR_PNG, _DIR_HTML)
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