# sim_runner.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

try:
    from tqdm import tqdm  
except ImportError: 
    def tqdm(it, **kwargs):  # type: ignore
        return it

import curation                                

import pandas as pd
from scenarios import load_scenarios_ecb, ScenarioECB
from multifirm_runner import run_ecb 

# helpers                                                                     #
def _collect(df: pd.DataFrame) -> pd.DataFrame:
    """Pass the raw output through Phase-E lightweight tidier."""
    return curation.tidy_dataframe(df)

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

    scenarios: List[ScenarioECB] = load_scenarios_ecb(args.config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    final_df = run_ecb(scenarios, jobs=args.jobs)
    final_df = _collect(final_df)

    final_df.to_parquet(args.out, index=False)

    # write a small head-preview for quick git-diffs
    preview = Path(args.out).with_suffix(".preview.csv")
    final_df.to_csv(preview, index=False)

    print(f"[sim_runner] ✅ wrote {len(final_df):,} rows -> {args.out}")
    print(f"[sim_runner] 📄 preview  -> {preview}")


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

A CSV lands next to it (same basename, .preview.csv extension) so pull requests show a human-readable diff instead of a binary blob.

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