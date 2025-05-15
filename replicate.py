# replicate.py
"""
One-shot Python front-end for replication

Equivalent to:  make reproduce
Run:           python replicate.py --jobs 4 --seed 42
"""
from __future__ import annotations
import argparse, sys, importlib

def _run(module_name: str, fn_name: str = "main", **kwargs) -> None:
    """Import `module_name` and invoke `fn_name(**kwargs)` if it exists."""
    mod = importlib.import_module(module_name)
    fn  = getattr(mod, fn_name, None)
    if callable(fn):
        print(f"[replicate] ➔ {module_name}.{fn_name}()")
        fn(**kwargs)
    else:
        raise AttributeError(f"{module_name} has no callable '{fn_name}'")

def main() -> None:
    p = argparse.ArgumentParser(description="Full replication runner (Phase H)")
    p.add_argument("--jobs", type=int, default=-1,
                   help="Parallel workers for sim_runner (-1 = all)")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional random seed for stochastic scenarios")
    args = p.parse_args()

    # 1 simulate  2 curate  3 plots  4 tables
    _run("sim_runner",          "main", argv=["--jobs", str(args.jobs)])
    _run("curation",            "curate")
    _run("visualise",           "render_all")
    _run("table_exporter",      "export")          # see next file
    print("\n[replicate] 🎉 Everything complete – artefacts ready.\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:      # pylint: disable=broad-except
        print(f"[replicate] ❌ {exc}", file=sys.stderr)
        sys.exit(1)


"""
1. What the script does
Parses two optional command-line flags

--jobs INT Number of CPU workers for the simulation stage (defaults to all cores with -1).

--seed INT Random seed reserved for future stochastic scenarios (currently not propagated downstream but kept for forward compatibility).

Sequentially invokes four pipeline stages by importing each helper module on-the-fly and calling a well-known function:

sim_runner.main() runs every ScenarioCfg, writes outputs/simulations.parquet (+ CSV preview).

curation.curate() validates and enriches the raw parquet, producing outputs/simulations_curated.parquet.

visualise.render_all() generates static PNG and interactive HTML plots into figures/ and figures_html/.

table_exporter.export() writes tables/summary_metrics.csv (and a LaTeX version if tabulate is installed).

2. Logs progress clearly
Before each stage the helper _run() prints something like

[replicate] ➔ sim_runner.main()
so you always know which component is running.

3. Fails fast and verbosely
Any un-caught exception inside a stage bubbles up; the script prints

[replicate]  <error message>
and returns a non-zero exit code so CI pipelines can flag the failure.

4. How to use it

# vanilla rebuild using every available core
python replicate.py

# restrict to four workers
python replicate.py --jobs 4

# fix the RNG for future Monte-Carlo scenarios
python replicate.py --seed 123

After a successful run you will find:

outputs/simulations_parquet raw long-form data

outputs/simulations_curated.parquet validated & enriched data

figures/*.png and figures_html/*.html publication-ready graphics

tables/summary_metrics.csv (+ optional .tex) key numeric summaries

No extra configuration files are needed; everything flows from the YAML definitions already in the repo.

"""