# plots/core.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Config
_PNG_DPI = 300
_DEFAULT_PALETTE = "Set2"        # colour-blind safe
_LOG = logging.getLogger(__name__)


# Helper: static matplotlib lineplot
def lineplot(
    df: pd.DataFrame,
    metric: str,
    spec: Dict[str, Any],
    outfile: Path,
) -> None:
    """
    Render a static PNG figure for the given metric.

    Parameters
    ----------
    df : pd.DataFrame
        Curated dataset (already schema-validated).
    metric : str
        Column to plot on the y-axis.
    spec : dict
        Plot specification from YAML (keys: title, ylabel, log_y).
    outfile : Path
        Destination PNG path (parent dir must exist).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for scenario, sub in df.groupby("scenario_id"):
        ax.plot(sub["t"], sub[metric], label=scenario)

    ax.set_title(spec["title"])
    ax.set_xlabel("Year t")
    ax.set_ylabel(spec["ylabel"])
    if spec.get("log_y"):
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize="small", ncol=2)
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=_PNG_DPI)
    plt.close(fig)
    _LOG.info("PNG written → %s", outfile.as_posix())


# Helper: interactive Plotly lineplot
def interactive_lineplot(
    df: pd.DataFrame,
    metric: str,
    spec: Dict[str, Any],
    outfile: Path,
) -> None:
    """
    Render an interactive HTML figure for the given metric.
    """
    fig = px.line(
        df,
        x="t",
        y=metric,
        color="scenario_id",
        title=spec["title"],
        labels={"t": "Year t", metric: spec["ylabel"]},
        log_y=spec.get("log_y", False),
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outfile, include_plotlyjs="cdn")
    _LOG.info("HTML written → %s", outfile.as_posix())


"""
1. What the module does

Central configuration
Two module-level constants define how every plot looks:

a) _PNG_DPI = 300 gives quality resolution for static images.

_DEFAULT_PALETTE = "Set2" keeps colours distinguishable for readers with colour-blindness.

b) lineplot() – static PNG generator

Receives the curated DataFrame, a metric name (e.g. "Y_new"), the YAML-derived spec, and a target Path.

Loops once over scenario_id, draws one line per scenario, sets the title/labels, optionally switches to a log scale, then writes the file.

Tight-layout, small legends, and automatic parent-directory creation are handled internally.

Every successful save is logged with logging.info, so CI logs tell you exactly which files were produced.

c) interactive_lineplot() – HTML generator

Wraps Plotly Express for a no-boilerplate interactive version of the same plot.

Uses the same metric and spec, and writes a self-contained HTML file (JS loaded from the Plotly CDN).

The colour palette matches the Matplotlib one, ensuring visual consistency between static and interactive outputs.

2. How to use it
You almost never call these functions directly—instead visualise.py iterates through plot_specs.yaml and dispatches:

from plots import lineplot, interactive_lineplot
lineplot(df, "y_growth_pct", spec_dict, Path("figures/y_growth_pct.png"))
interactive_lineplot(df, "y_growth_pct", spec_dict, Path("figures_html/y_growth_pct.html"))
The only requirements are:

Validated data – The DataFrame should have already passed validator.SCHEMA.

A spec dictionary – Must contain at least title and ylabel; optional log_y: true toggles a log axis.

3. Customisation points
Changing DPI or palette – Edit _PNG_DPI or _DEFAULT_PALETTE at the top of the file and every downstream plot will adopt the new setting.

Additional plot types – If you need bar charts or heatmaps later, add a new helper (e.g. barplot()) here; visualise.py will pick it up once you reference it in plot_specs.yaml.

Alternate colour themes – Swap px.colors.qualitative.Set2 for any Plotly qualitative palette and keep the same name in the Matplotlib branch for coherence.
"""