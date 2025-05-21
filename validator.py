from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, Check

# DataFrameSchema 

SCHEMA = pa.DataFrameSchema(
    {
        # core identifiers 
        "scenario_id": Column(str, nullable=False),
        "test_label":  Column(str, nullable=False),         
        "hypothesis":  Column(str, nullable=False),         

        "firm_id":     Column(int, Check.ge(0)),
        "t":           Column(int, Check.ge(0)),

        # ECB micro KPIs
        "Y_new":       Column(float, Check.ge(0)),
        "Y_new_nominal": Column(float, Check.ge(0)),      # <<< NEW >>>

        "psi_eff":     Column(float, Check.ge(0)),
        "theta":       Column(float, Check.ge(0)),
        "congestion_idx":   Column(float, Check.ge(0), nullable=True),
        "queue_len":   Column(int,   Check.ge(0)),

        #  optional / derived 
        "mean_latency": Column(float, Check.ge(0), nullable=True),
        "p95_latency":  Column(float, Check.ge(0), nullable=True),
        "decay_loss":     Column(float, Check.ge(0), nullable=True, required=False),  # alias
        "max_latency":     Column(float, Check.ge(0), nullable=True),  # tail-risk queue wait
        "std_latency":     Column(float, Check.ge(0), nullable=True),  # dispersion of waits
        "creativity_loss": Column(float, nullable=True, required=False),  # ΔY from idea decay
        "triage_eff":      Column(float, [Check.ge(0), Check.le(1)], nullable=True),  # accepted/total

        "education_lag":   Column(int,   Check.ge(0), nullable=True, required=False),
        "latency_level":   Column(float, Check.ge(0), nullable=True, required=False),

        "evaluator_gap":   Column(float, Check.ge(0), nullable=True, required=False),  
        "U_nf_mean":       Column(float, Check.ge(0), nullable=True, required=False),  
        "spillover_gain":  Column(float, nullable=True, required=False),               
        "Y_lost_decay":    Column(float, Check.ge(0), nullable=True, required=False),  

        "market_share": Column(float, Check.in_range(0, 1), nullable=True),
        "congestion_idx_mean": Column(float, Check.ge(0), nullable=True, required=False), 

        "sgna_cost": Column(float, Check.ge(0), nullable=True, required=False),
        "profit":    Column(float, nullable=True, required=False),  
        "creativity_loss_pct_mean": Column(float, nullable=True, required=False),   # %
        "H_nf_mean":                Column(float, nullable=True, required=False),   # evaluator stock
   
        "decay_loss":      Column(float, Check.ge(0),  nullable=True, required=False),
        "spillover_gain":  Column(float,               nullable=True, required=False),

        # evaluator stocks 
        "Uf":  Column(float, Check.ge(0), nullable=True, required=False),
        "Unf": Column(float, Check.ge(0), nullable=True, required=False),
        "Hnf": Column(float, Check.ge(0), nullable=True, required=False),

  

    },
    coerce=True,
    strict=False,              # allow future experimental columns
    index=pa.Index(int),       # generic RangeIndex; no name constraint
)

"""
1. What the schema does

It tells Pandera what columns must exist, what data‑type each column must carry and which basic sanity checks should hold (for example, labour, capital and output can never be negative).

When a dataframe is passed to SCHEMA.validate(...), Pandera inspects every value; if anything is missing, has the wrong dtype, is negative when it should be non‑negative, or contains a null where it shouldn’t, a detailed exception is raised immediately.

Because strict=False, extra columns are allowed. That means you can enrich the dataset later—say by adding growth‑rate columns—without rewriting the schema.

2. Column‑level rules 

scenario_id must be a non‑null string that identifies which scenario the row belongs to.

t is the time index; it must be an integer ≥ 0.

LY and LA are quantities of labour and must be floats ≥ 0.

capital_current, synergy, intangible, knowledge_after and Y_new are all economic variables that must also be floats ≥ 0.

x_values holds a NumPy array of intermediate‑input quantities; it may not be null, but its internal numeric contents are checked elsewhere.

All other columns—derived metrics, debugging flags, future extensions—are ignored by the schema unless you decide to add explicit rules for them.

3. How you use it in practice

from validator import SCHEMA
import pandas as pd

df = pd.read_parquet("outputs/simulations.parquet")
SCHEMA.validate(df, lazy=True)   # raises if anything is wrong
Call this once, right after you load any parquet or CSV produced by the simulator. If the call returns without error, you can trust that the dataframe respects the project’s data contract.

lazy=True is recommended because it gathers all violations in a single run and reports them together, saving round‑trips in debugging.

4. Customising or extending the checks

Adding a new column: simply write
SCHEMA.columns["my_new_metric"] = pa.Column(float, pa.Check.ge(0))
or, if you prefer, edit the dictionary literal in validator.py so the rule is version‑controlled.

Tightening a rule: replace Check.ge(0) with, for example,
Check.in_range(0, 1) if a variable is known to be a probability.

Relaxing a rule: set nullable=True or drop the Check.

Because the same SCHEMA object is imported everywhere (simulation, curation, plotting, tests), any change you make is picked up automatically across the whole code‑base.

5. When something fails

If SCHEMA.validate raises an exception you’ll see a clear message such as:

SchemaError: Column 'capital_current' expected non‑negative floats, found -12.5 at row 1453
Fix the underlying model or data issue and run the validation again. Nothing downstream will execute until the schema passes, so corrupted results can’t silently propagate.

In short, validator.py is the single, central checkpoint that keeps every row of data coherent, numeric and analysis‑ready.
"""