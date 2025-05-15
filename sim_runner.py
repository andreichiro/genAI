# sim_runner.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

from joblib import Parallel, delayed
try:
    from tqdm import tqdm  
except ImportError: 
    def tqdm(it, **kwargs):  # type: ignore
        return it

from scenarios import load_scenarios, ScenarioCfg
from genAI import compute_y_romer, GrowthExplosionError  
import curation                                

# helpers                                                                     #

def _run_single(cfg: ScenarioCfg) -> pd.DataFrame:
    """Run one scenario and return a tidy DataFrame of step logs."""
    try:
        res: Dict[str, Any] = compute_y_romer(
            num_periods=cfg.num_periods,
            alpha=cfg.alpha,
            labor_func=cfg.labor_func,
            capital_func=cfg.capital_func,
            synergy_func=cfg.synergy_func,
            intangible_func=cfg.intangible_func,
            x_values_updater=cfg.x_values_updater,
            total_labor_func=cfg.total_labor_func,
            share_for_rd_func=cfg.share_for_rd_func,
            dt_integration=cfg.dt_integration,
            growth_explosion_threshold=cfg.growth_explosion_threshold,
            intangible_kappa=cfg.intangible_kappa,
            intangible_Ubar=cfg.intangible_Ubar,
            intangible_epsilon=cfg.intangible_epsilon,
            store_results=True,
        )
    except GrowthExplosionError as exc:            # +++ NEW +++
        raise RuntimeError(f"[sim_runner] {exc}") from exc

    if res["growth_explosion"]:
        raise RuntimeError(f"[sim_runner] Growth explosion in scenario '{cfg.id}'")

    df = pd.DataFrame(res["outputs"])
    df["scenario_id"] = cfg.id
    return df


def _collect(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate & lightly coerce via Phase-B curation stub."""
    long = pd.concat(dfs, ignore_index=True)
    return curation.tidy_dataframe(long)

# CLI                                                                         #

def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run all scenario simulations.")
    p.add_argument("--config", default="scenarios.yaml",
                   help="Path to YAML with scenario definitions")
    p.add_argument("--out", default="outputs/simulations.parquet",
                   help="Destination Parquet file")
    p.add_argument("--jobs", type=int, default=-1,
                   help="Parallel workers (-1 = all cores, 1 = sequential)")

    args = p.parse_args(argv)

    scenarios = load_scenarios(args.config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.jobs == 1:
        dfs = [_run_single(cfg) for cfg in tqdm(scenarios, desc="Simulations")]
    else:                                 # parallel branch
        dfs = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(_run_single)(cfg) for cfg in tqdm(scenarios, desc="Simulations")
        )

    final_df = _collect(dfs)
    final_df.to_parquet(args.out, index=False)

    # write a small head-preview for quick git-diffs
    preview = Path(args.out).with_suffix(".preview.csv")
    final_df.head(10).to_csv(preview, index=False)

    print(f"[sim_runner] ✅ wrote {len(final_df):,} rows -> {args.out}")
    print(f"[sim_runner] 📄 preview (10 rows) -> {preview}")


if __name__ == "__main__":                # entry-point
    try:
        main()
    except RuntimeError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

"""
1. What the script does, step by step

Parse CLI arguments

--config – path to the YAML file that lists scenarios (default scenarios.yaml).

--out – where to save the aggregated results (default outputs/simulations.parquet).

--jobs – number of parallel worker processes. -1 uses all CPU cores; 1 disables parallelism for deterministic debugging.

2. Load scenarios
Calls load_scenarios() from scenarios.py; this returns a list of fully-typed ScenarioCfg objects containing every callable the solver needs.

3. Run each scenario

In sequential mode (--jobs 1) it loops directly.

In parallel mode it uses joblib.Parallel with the rock-solid “loky” backend, so each worker lives in its own process and Python’s GIL is not a bottleneck.

4. Perform a full model simulation
Each worker invokes _run_single, which simply passes the scenario’s callables into compute_y_romer (the heavy model in genAI.py).
A scenario that triggers a numerical growth explosion raises GrowthExplosionError; _run_single catches it and re-raises a clear, scenario-specific RuntimeError so the overall run aborts quickly and noisily.

5. Collect results
The per-scenario DataFrames are concatenated, then run through curation.tidy_dataframe for lightweight type coercion and column ordering. No heavy validation happens here—that’s Phase C’s job.

6. Write artefacts

The full long-format dataset becomes outputs/simulations.parquet.

A tiny 10-row preview CSV lands next to it (same basename, .preview.csv extension) so pull requests show a human-readable diff instead of a binary blob.

7. Exit codes and messages
A friendly ✓ summary prints on success. On any fatal error (e.g., growth explosion) the script writes the message to stderr and exits with non-zero status so CI will fail.

8. How to run it

# vanilla run on all cores
python -m sim_runner

# deterministic single-thread run, custom locations
python -m sim_runner --jobs 1 \
                     --config configs/my_scenarios.yaml \
                     --out    artefacts/my_run.parquet

9. Configuration knobs at a glance

Parallelism – --jobs flag; give a positive integer to pin to that many cores.

Scenario set – point --config to any YAML file that follows the schema used by scenarios.yaml; no code edits needed.

Output location – --out; directories are created automatically if missing.

Progress bar dependency – If tqdm is not installed the script falls back to a no-op wrapper, so it never crashes on minimal environments.

10. Internal design choices that matter

Pure functions, easy pickling – Every scenario’s callables are defined at module scope, so joblib can pickle them without hacks.

Fail-fast philosophy – The very first numerical blow-up aborts the whole batch; you’ll never wait minutes just to discover post-mortem that half your runs diverged.

No hard-coded parameters – All economics numbers live in YAML; the Python merely orchestrates.
"""