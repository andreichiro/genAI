# table_exporter.py
"""
helper to produce summary tables from `simulations_curated.parquet`.

• CSV written to tables/summary_metrics.csv
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import tabulate                          # lightweight dep

_CUR   = Path(__file__).parent
_DATA  = _CUR / "outputs" / "simulations_curated.parquet"
_TDIR  = _CUR / "tables"
_TCSV  = _TDIR / "summary_metrics.csv"
_TTEX  = _TDIR / "summary_metrics.tex"

# Meta & metric taxonomy --------------------------------------------------------
_META = ["scenario_id", "test_label", "hypothesis"]
_BASE     = ["Y_new", "y_growth_pct", "capital_intensity", "rd_share"]
_LATENCY  = ["mean_latency", "p95_latency", "max_latency", "std_latency"]
_MARKET   = ["market_share", "congestion_idx_mean"]
_DECAY    = ["creativity_loss", "Y_lost_decay"]

# master list, preserving section order; duplicates automatically removed
_ALL_METRICS: list[str] = []
for _block in (_BASE, _LATENCY, _MARKET, _DECAY):
   _ALL_METRICS.extend([m for m in _block if m not in _ALL_METRICS])

def _build_summary(df: pd.DataFrame) -> pd.DataFrame:            # ← sig unchanged
    """
    Scenario-level (actually meta-level) mean of **every** numeric KPI.

    *Primary list* (_ALL_METRICS) is shown first; any additional numeric
    columns that slipped through curation are appended automatically so
    the table never “loses” information when KPIs evolve.
    """
    gb = df.groupby(_META, dropna=False)

    # 1️⃣ mean aggregation for declared metrics ---------------------- ### NEW ###
    summary = gb[_ALL_METRICS].mean()

    # 2️⃣ pick up any extra numeric columns not yet covered ---------- ### NEW ###
    extra_cols = (
        df.select_dtypes(include="number")
          .columns.difference(summary.columns, sort=False)
    )
    if len(extra_cols):
        summary = summary.join(gb[extra_cols].mean())

    return (summary
              .reset_index()
              .sort_values(_META))


def export() -> None:
    _TDIR.mkdir(exist_ok=True)
    # load *all* columns; _build_summary will filter/aggregate as needed
    df = pd.read_parquet(_DATA)
    summary = _build_summary(df)
    summary.to_csv(_TCSV, index=False) 
    # quick LaTeX (booktabs) for journals – user may ignore if not needed
    try:
        import tabulate                          # lightweight dep
        latex = tabulate.tabulate(summary, headers="keys",
                                  tablefmt="latex_booktabs",
                                  floatfmt=".4f")
        _TTEX.write_text(latex)
    except ModuleNotFoundError:
        print("[table_exporter] tabulate not installed – TeX skipped")

    print(f"[table_exporter] 📊 CSV  → {_TCSV}")
    if _TTEX.exists():
        print(f"[table_exporter] 📄 LaTeX→ {_TTEX}")

if __name__ == "__main__":
    export()


"""
1. What the script generates

tables/summary_metrics.csv – a comma-separated file in which each row is a scenario and each column is the arithmetic mean of a key metric across all time-steps.

tables/summary_metrics.tex – an optional LaTeX booktabs table containing the same numbers, ready to drop into a journal manuscript. The LaTeX file is written only if the lightweight tabulate package is available; otherwise the script prints a friendly notice and skips that step.

2. Metrics included by default
The list _METRICS = ["Y_new", "y_growth_pct", "capital_intensity", "rd_share"] defines which variables are averaged.

Y_new – raw model output (level).

y_growth_pct – period-over-period percentage growth, added during the curation phase.

capital_intensity – capital-to-labour ratio.

rd_share – fraction of labour devoted to R&D.

You can change the mix by editing _METRICS; no other code changes are required.

3. How it works internally
Locate the curated parquet (outputs/simulations_curated.parquet) and load only the columns required for the calculation, saving memory.

Group by scenario_id and compute the mean of each metric – this collapses hundreds of time-steps into one row per scenario.

Write results to CSV so any spreadsheet program can open them.

Optionally emit LaTeX via the tabulate library, producing a nicely formatted table that honours booktabs styling conventions. If tabulate is missing the script still succeeds, it simply omits the TeX file.

Console feedback tells you exactly where each file landed.

4. Running the exporter

python -m table_exporter        # or:  python table_exporter.py
You will see something like:


[table_exporter] 📊 CSV  → /path/to/project/tables/summary_metrics.csv
[table_exporter] 📄 LaTeX→ /path/to/project/tables/summary_metrics.tex
If LaTeX output is suppressed (because tabulate is not installed) you will instead see:


[table_exporter] tabulate not installed – TeX skipped

5. Configuration knobs you might tweak

Change output directory – edit _TDIR at the top of the file if you prefer a different location than tables/.

Switch to median or another aggregation – replace .mean() in _build_summary with .median() or any other pandas aggregator.

Skip LaTeX entirely – comment out or delete the tabulate block.

Add or drop metrics – adjust the _METRICS list; the grouping logic will pick up whatever you specify automatically.

The script has no command-line arguments because its purpose is intentionally narrow and the configuration lives in easily editable constants at the top of the file.

6. Typical workflow
Run the full simulation and curation pipeline (produces simulations_curated.parquet).

Execute python -m table_exporter.

Open tables/summary_metrics.csv in Excel or include tables/summary_metrics.tex in your LaTeX document.
"""