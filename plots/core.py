# plots/core.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

import numpy as np
import textwrap
from utils.screening_shared import congestion_tail, psi_peak_series

# Config
_PNG_DPI = 300
_DEFAULT_PALETTE = "Set2"        # colour-blind safe
_LOG = logging.getLogger(__name__)
_META_COLS = ["scenario_id", "test_label", "hypothesis"]      

def _wrap_label(txt: str, width: int = 25) -> str:            # <<< NEW >>>
    """Soft-wrap *txt* so long legends never spill outside the figure."""
    return "\n".join(textwrap.wrap(str(txt), width))

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
    # ensure the static Matplotlib palette matches the interactive
    # Plotly palette (`Set2`) so colours stay consistent across outputs 
    # and remain colour-blind friendly
    ax.set_prop_cycle(color=plt.get_cmap(_DEFAULT_PALETTE).colors)

    for scenario, sub in df.groupby("scenario_id"):
            # Compose “scenario | label | hypothesis” once per line
            meta_label = _wrap_label(
                " | ".join(map(str, sub.iloc[0][_META_COLS]))          
            )
            ax.plot(sub["t"], sub[metric], label=meta_label)    

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

    df = df.assign(                                                  # <<< NEW >>>
        _label=lambda d: d[_META_COLS]
                .astype(str)
                .agg(" | ".join, axis=1)
                .map(_wrap_label)
    )

    fig = px.line(
        df,
        x="t",
        y=metric,
        color="_label",                                              # <<< CHG >>>
         title=spec["title"],
         labels={"t": "Year t", metric: spec["ylabel"]},
         log_y=spec.get("log_y", False),
         color_discrete_sequence=px.colors.qualitative.Set2,
     )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outfile, include_plotlyjs="cdn")
    _LOG.info("HTML written → %s", outfile.as_posix())

def tail_cdf_plot(U_bar_series, *, ax=None,
                  title="Congestion tail Φ(u)"):
    """
    Plot the empirical survival CDF Φ(u) for the mean-field series Ū(t).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    phi = congestion_tail(U_bar_series)
    ax.plot(phi.index, phi.values, linewidth=2)
    ax.set_xlabel("u  (mean evaluation capital)")
    ax.set_ylabel("Φ(u) = P[Ū ≥ u]")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    return ax


def psi_peak_plot(df_firms, *, kpi_col="psi_eff",
                  ax=None, title="ψ_eff peak diagnostic"):
    """
    For every period t, plot the maximum ψ_eff across firms.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    peak = psi_peak_series(df_firms, kpi_col=kpi_col)
    ax.plot(peak.index, peak.values, linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("max ψ_eff(t)")
    ax.set_title(title)
    return ax


def histplot(df: pd.DataFrame,
             metric: str,
             spec: dict,
             outfile: Path) -> None:
    """
    Static histogram (Matplotlib) for a scalar metric – used for
    σ̂² distribution, Creativity_loss, Triage_eff, etc.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[metric].dropna(), bins=spec.get("bins", 40))
    ax.set_xlabel(spec["ylabel"])
    ax.set_ylabel("Count")
    ax.set_title(spec["title"])
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=_PNG_DPI)
    plt.close(fig)
    _LOG.info("Histogram written → %s", outfile.as_posix())

def barplot(df: pd.DataFrame,
            metric: str,
            spec: dict,
            outfile: Path) -> None:
    """
    Scenario-level bar chart – complements histograms for metrics where
    cross-scenario comparison is more useful than a distribution view.
    Intended for KPI means such as ROI_skill (Phase G tables & plots).

    Parameters
    ----------
    df : pd.DataFrame
        Curated dataset.
    metric : str
        Column name to aggregate (mean) then plot.
    spec : dict
        Plot spec from YAML (keys: title, ylabel, optionally rotate_xticks).
    outfile : Path
        Where the PNG should be written.
    """
    means = (df.groupby("scenario_id")[metric]
               .mean()
               .sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(means.index, means.values)
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])

    if spec.get("rotate_xticks", True):
        ax.set_xticklabels(
            [_wrap_label(idx, 20) for idx in means.index],        
            rotation=45,
            ha="right",
        )

    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=_PNG_DPI)
    plt.close(fig)
    _LOG.info("Bar chart written → %s", outfile.as_posix())


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