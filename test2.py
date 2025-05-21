import pandas as pd, pyarrow.parquet as pq, pathlib
cur = pathlib.Path("outputs/simulations_curated.parquet")
print(cur.exists(), pq.ParquetFile(cur).metadata.num_rows)
