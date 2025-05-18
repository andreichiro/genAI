# multifirm_runner.py
# Mean-field congestion layer for the ECB refactor
#
# Key responsibilities
# --------------------
# 1.  Advance every firm one period, *then* compute the cross-firm mean
#     Ū_{-i,t}.  That value feeds back into `screening_capacity()` the
#     *next* period, closing the congestion loop without introducing
#     simultaneity problems.
# 2.  Persist per-period diagnostics — especially the time–series of
#     U_bar (mean evaluation capital) and its empirical CDF tails —
#     so Phase E4 can plot “ψ-peak” and Φ(t) curves without re-running
#     the heavy simulation.
# 3.  Remain agnostic about firm-level micro-dynamics.  We delegate one
#     function pointer `firm_step(state, U_external_prev)` that mutates a
#     dataclass `FirmECBState` in-place and returns a dict of KPIs.
#
# The runner never imports private internals from the solver: it only
# needs public helpers   `screening_utils.screening_capacity`   and the
# dataclasses `ECBParams` + `TriageParams`, both already exported.
# --------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence
import numpy as np
import pandas as pd
from utils.ecb_params import ECBParams
from utils.mobility_utils import poach_evaluators
from collections import deque 
from utils.edu_updater    import update_supply    # ← NEW

from joblib import Parallel, delayed
try:                                        # progress bar is nice but optional
    from tqdm import tqdm
except ImportError:                         # pragma: no cover
    def tqdm(it, **_kwargs):                # type: ignore
        return it

from utils.ecb_firm_step import ecb_firm_step     
from scenarios import ScenarioECB
from utils.queue_dynamics import new_queue 

__all__ = ["run_ecb", "FirmECBState", "run_mean_field_sim"]

# 1 ▸ *Per-firm* mutable container updated each period
@dataclass(slots=True)
class FirmECBState:
    """Minimal state needed for the congestion externality."""
    firm_id:  int
    Uf:       float
    Unf:      float
    Hnf:      float
    K_AI:   float                     # Poisson intensity scale ▲
    mu_prior: float = 0.0 
    queue:    deque        = field(default_factory=new_queue)
    alpha:    float = 0.33 
    L_Y:      float = 1.0       
    x_values: list[float] = field(default_factory=lambda: [1.0])
    queue_len:int      = 0   # optional – maintained by queue_dynamics
    # attach the scenario’s immutable parameter bundle
    ecb:      ECBParams = ECBParams()
    # Convenient derived — updated by `update_U_tot()` each tick
    U_tot:    float     = 0.0

    # -----------------------------------------------------------------
    def update_U_tot(self) -> None:
        """Re-compute U_tot from the three capital stocks."""
        self.U_tot = (
            self.Uf
            + self.ecb.xi1 * self.Unf * (self.Hnf ** self.ecb.zeta_skill_exp)
        )


# 2 Signature of the user-supplied per-period micro step
StepFunc = Callable[[int, FirmECBState, float], Dict[str, float]]
# The float argument is U_external_prev (mean capital previous period).
# The dict it returns will be concatenated into a long DataFrame.

def _make_states(firms_init: Sequence[Dict[str, float]],
                 ecb: ECBParams) -> List[FirmECBState]:
    states: List[FirmECBState] = []
    for idx, spec in enumerate(firms_init):
        states.append(
            FirmECBState(
                firm_id   = idx,                 # maintain order -> deterministic
                Uf        = spec["U_f"],
                Unf       = spec["U_nf"],
                Hnf       = spec["H_nf"],
                K_AI    = spec.get("K_AI",  1.0),        # ▲ default 1
                queue   = new_queue(),           
                ecb       = ecb,
            )
        )
        states[-1].update_U_tot()               # prime the derived field
    return states

# 3 Main driver
def run_mean_field_sim(
    *,
    firm_states: Sequence[FirmECBState],
    step_func:  StepFunc,
    num_periods: int,
    mobility_elasticity: float = 0.0,
    shared_pool: bool = False,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Simulate a *finite* collection of firms under a mean-field
    congestion externality.

    Parameters
    ----------
    firm_states : sequence[FirmECBState]
        Initial conditions for every firm.
    step_func : StepFunc
        User-supplied micro-update.
    num_periods : int
        Horizon T.
    out_csv : pathlib.Path, optional
        If given, results are also written to CSV for quick inspection.

    Returns
    -------
    pandas.DataFrame
        Long-format table (firm_id, t, KPI columns …).
    """
    if num_periods <= 0:
        raise ValueError("num_periods must be positive")

    # storage for diagnostics 
    rows: List[Dict[str, float]] = []
    
    p_ecb = firm_states[0].ecb                    # same params for all firms
    edu_lag = max(int(p_ecb.education_lag), 0)

    if edu_lag > 0:                             
        trainee_pipe = deque([0.0] * edu_lag, maxlen=edu_lag)
    else:
        trainee_pipe = deque()                  # harmless placeholder

    trainee_stock: float = p_ecb.initial_trainees
    trainee_pipe  = deque([0.0] * edu_lag, maxlen=edu_lag)
    # initialise “external” capital metric used inside screening_capacity
    if shared_pool:
        U_external_series: List[float] = [float(np.sum([f.U_tot for f in firm_states]))]  # t = −1
    else:
        U_external_series: List[float] = [float(np.mean([f.U_tot for f in firm_states]))]  # t = −1

    # main loop 
    for t in range(num_periods):
        U_external_prev = U_external_series[-1]

        period_rows: List[Dict[str, float]] = []        # buffer to add market_share
        wages: Dict[int, float] = {}                    # for evaluator mobility
        Y_vals: List[float] = []                        # collect for ΣY

        for state in firm_states:
            kpi = step_func(t, state, U_external_prev)    
            I_train = kpi.get("I_train")            # ← NEW optional KPI
            if I_train:
                trainee_stock += float(I_train)

            state.update_U_tot()                        # refresh derived
            wage_i = kpi.get("wage")                    # optional
            if wage_i is not None:
                wages[state.firm_id] = float(wage_i)
            Y_now = kpi.get("Y_new")
            if Y_now is not None:
                Y_vals.append(float(Y_now))
            period_rows.append({"firm_id": state.firm_id, "t": t, **kpi})

        # evaluator mobility (strategic poaching) – only if ε>0
        if mobility_elasticity > 0.0 and wages:
            poach_evaluators(firm_states, wages, mobility_elasticity)

        # 2️⃣  market-share post-processing
        Y_tot = sum(Y_vals)
        for row in period_rows:
            Y_i = row.get("Y_new")
            row["market_share"] = (Y_i / Y_tot) if (Y_i is not None and Y_tot > 0) else np.nan
            rows.append(row)


        # 2️⃣  compute *current* mean for next period’s call

        if shared_pool:
            U_external_now = float(np.sum([f.U_tot for f in firm_states]))
        else:
            U_external_now = float(np.mean([f.U_tot for f in firm_states]))
        U_external_series.append(U_external_now)

        # update trainee pipeline  ➜ inject graduates equally
        if edu_lag > 0:
            trainee_stock, grads = update_supply(
                S_prev=trainee_stock,
                pipeline=trainee_pipe,
                p=p_ecb,
            )
            if grads > 0.0:
                add_each = grads / len(firm_states)
                for st in firm_states:
                    st.Unf += add_each

        logging.debug(
                "t=%d  shared_pool=%s  U_prev=%.3f  U_now=%.3f",
                t, shared_pool, U_external_prev, U_external_now
            )
    
    # tail diagnostics 
    tail95 = np.percentile(U_external_series, 95)
    logging.info("Φ(t) tail-95 = %.3f  (max %.3f)",
                 tail95, max(U_external_series))

    # tidy output 
    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logging.info("multifirm_runner ▸ wrote %s  (%d rows)",
                     out_csv.as_posix(), len(df))
    return df


def _run_single_scenario(scn) -> pd.DataFrame:            # ScenarioECB
    """Run one ScenarioECB through the mean-field engine and tag rows."""
    states = _make_states(scn.firms_init, scn.ecb_params)

    rng = np.random.default_rng(hash(scn.id) & 0xFFFFFFFF)

    # wrap the Phase-B micro-step so it sees the scenario’s RNG & triage knobs
    def _step(t_tick: int, state: FirmECBState, U_ext_prev: float):
        return ecb_firm_step(
            t               = t_tick,
            state           = state,
            params          = scn.ecb_params,
            U_bar_others    = U_ext_prev,
            rng             = rng,
            triage_params   = scn.triage_params,
        )
    df = run_mean_field_sim(
        firm_states        = states,
        step_func          = _step,
        num_periods        = scn.num_periods,
        mobility_elasticity= scn.ecb_params.mobility_elasticity,
        shared_pool        = scn.ecb_params.shared_pool,
    )
    df["scenario_id"] = scn.id
    return df

def run_ecb(
    scenarios: Sequence["ScenarioECB"],        # string literal → avoids import cycle
    *,
    jobs: int = -1,
) -> pd.DataFrame:
    """
    Phase-D convenience wrapper used by sim_runner.py.

    Dispatches every ScenarioECB to the mean-field engine and concatenates
    the long-format outputs.  Parallelism is fully optional.
    """
    if jobs == 1:
        dfs = [_run_single_scenario(s) for s in tqdm(scenarios, desc="ECB runs")]
    else:
        dfs = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_run_single_scenario)(s) for s in tqdm(scenarios, desc="ECB runs")
        )
    return pd.concat(dfs, ignore_index=True)
