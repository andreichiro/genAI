import pandas as pd, yaml, pathlib, textwrap, numpy as np

DATA = pathlib.Path("outputs/simulations.parquet")
if not DATA.exists():
    raise SystemExit(f"❌ {DATA} not found – Phase C never finished?")

df     = pd.read_parquet(DATA)
specs  = yaml.safe_load(open("plot_specs.yaml"))
report = []

for metric, spec in specs.items():
    if metric not in df.columns:
        report.append((metric, "❌  missing"))
    else:
        n_finite = df[metric].notna().sum()
        if n_finite == 0:
            report.append((metric, "⚠️  all-NaN"))
        else:
            report.append((metric, f"✓  {n_finite:,} values"))

col_w  = max(len(k) for k,_ in report)+2
for k,msg in report:
    print(f"{k:<{col_w}} {msg}")
