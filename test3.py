import pandas as pd, pathlib, yaml

DATA = pathlib.Path("outputs/simulations_curated.parquet")
print("exists ->", DATA.exists())

df = pd.read_parquet(DATA)
specs = yaml.safe_load(open("plot_specs.yaml"))

for m in specs:
    print(f"{m:22s}", "present" if m in df.columns else "MISSING")
