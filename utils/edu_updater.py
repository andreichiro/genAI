"""
edu_updater.py
────────────────────────────────────────────────────────────────────────────
Sector-wide supply pipeline for *non-fungible* human evaluators (H_nf).

We keep a stock S(t) of **trainees currently in the pipeline**.
Each period:
    • A fraction `enroll_rate` of S(t) starts the training clock.
    • Trainees graduate and become productive after `grad_lag` periods.
    • A small fraction `retire_rate` of seasoned evaluators leaves.

The helper below returns   (S_new, grads_this_period)
so the caller (multifirm_runner) can inject fresh graduates into firms’
U_nf stock.

All parameters come from ECBParams so caller code stays minimal.
"""
from __future__ import annotations

from collections import deque
from typing import Tuple

from utils.ecb_params import ECBParams

def new_pipeline(p: ECBParams) -> deque[float]:                           # ← [NEW]
    """Return a deque of length `p.education_lag` filled with zeros."""   # ← [NEW]
    return deque([0.0] * p.education_lag, maxlen=p.education_lag)        # ← [NEW]


def update_supply(
    *,
    S_prev: float,
    pipeline: "deque[float]",
    p: ECBParams,
) -> Tuple[float, float]:
    """
    Advance the education pipeline one period.

    Parameters
    ----------
    S_prev : float
        Trainee stock *entering* the period.
    pipeline : collections.deque
        Length exactly `p.education_lag`.  pipeline[i] holds the cohort
        enrolled **i** periods ago but not yet graduated.
    p : ECBParams
        Must provide education_lag, enroll_rate, retire_rate.

    Returns
    -------
    (S_new, grads) : tuple[float, float]
        • S_new   – trainee stock after enrolment/retirement
        • grads   – head-count of newly graduated evaluators
    """
    if S_prev < 0:
        raise ValueError("S_prev must be ≥ 0")
    if len(pipeline) != p.education_lag:                                  
        raise ValueError("`pipeline` length must equal p.education_lag")  

    # 1. graduates leave the right-hand end
    if p.education_lag > 0:                                            
        grads = pipeline.pop()                                          
    else:                                                            
        grads = 0.0                                                    

    new_enrol = p.enroll_rate * S_prev + p.enroll_const
    if p.education_lag > 0:                                            
        pipeline.appendleft(new_enrol)                                  

    # 3. trainee stock evolves with enrolment – retirement
    S_after_enrol = S_prev - grads + new_enrol
    S_new = (1.0 - p.retire_rate) * S_after_enrol

    return S_new, grads
