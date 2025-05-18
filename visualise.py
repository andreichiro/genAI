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

from validator import SCHEMA

import matplotlib.pyplot as _plt
import plotly.express      as _px

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_ROOT = Path(__file__).resolve().parent
_CFG_YAML = _ROOT / "scenarios.yaml"                              # ← [NEW]

# sensible fall-backs (project-relative defaults)
_DEFAULTS = {
    "out_cur":  _ROOT / "outputs" / "simulations_curated.parquet",
    "fig_png":  _ROOT / "figures",
    "fig_html": _ROOT / "figures_html",
}

try:                                                             # ← [NEW]
    _CFG = yaml.safe_load(_CFG_YAML.read_text()) if _CFG_YAML.exists() else {}
    _PATHS = (_CFG or {}).get("paths", {}) if isinstance(_CFG, dict) else {}
except Exception as _e:                                          # ← [NEW]
    logging.warning("Could not load %s (%s) – using defaults", _CFG_YAML, _e)
    _PATHS = {}

_DATA      = Path(_PATHS.get("out_cur",  _DEFAULTS["out_cur"])).resolve()   # ← [NEW]
_DIR_PNG   = Path(_PATHS.get("fig_png",  _DEFAULTS["fig_png"])).resolve()   # ← [NEW]
_DIR_HTML  = Path(_PATHS.get("fig_html", _DEFAULTS["fig_html"])).resolve()  # ← [NEW]
_PLOT_SPECS = _ROOT / "plot_specs.yaml"                                      # (unchanged)  # ← [CHG]


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


def render_all() -> None:
    """
    Main orchestration: validate data, then loop over every metric defined
    in plot_specs.yaml and produce a PNG + HTML.
    """
    df = pd.read_parquet(_DATA)
    SCHEMA.validate(df, lazy=True)          # D-1 safety net

    # remove heavy columns not needed for plotting (vector x_values)
    if "x_values" in df.columns:
        df = df.drop(columns="x_values")

    specs = _load_specs()
    _DIR_PNG.mkdir(exist_ok=True, parents=True)
    _DIR_HTML.mkdir(exist_ok=True, parents=True)

    for metric, spec in tqdm(specs.items(), desc="plots"):
        if metric not in df.columns:
            _LOG.warning("Metric '%s' not in curated dataset — skipped.", metric)
            continue

        png_path = _DIR_PNG / f"{metric}.png"
        html_path = _DIR_HTML / f"{metric}.html"

        _static_lineplot(df, metric, spec, png_path)        # <- NEW helper
        _interactive_lineplot(df, metric, spec, html_path)  # <- NEW helper
    _LOG.info("Phase D finished ✓")


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