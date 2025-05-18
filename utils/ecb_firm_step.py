# ecb_firm_step.py

# Single-period micro-dynamics for one firm under the Evaluator-Centric
# theory.  All heavy math is delegated to helper modules inside utils/.

from __future__ import annotations

import numpy as np
from typing import Dict, Any

# helper layers (implemented in the next files we add/patch)
from .triage_utils   import bayes_posterior, triage_score  
from .aggregator_ecb  import output_ecb
from .ecb_params      import ECBParams
from .screening_utils  import psi_efficiency, theta_accuracy

try:                                    # preferred: package-relative
    from .queue_dynamics import enqueue_new_ideas, service_queue_fifo
except ImportError:                     # fallback: absolute
    from utils.queue_dynamics import enqueue_new_ideas, service_queue_fifo

def ecb_firm_step(
    *,
    t: int,
    state: "FirmECBState",
    params: ECBParams,
    U_bar_others: float,
    triage_params=None,  # Add this parameter with a default value
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Advance one firm by a single period and return a tidy KPI dict.
    The caller (multifirm_runner) is responsible for row assembly.
    """
    # 1) Idea arrivals -------------------------------------------------------
    num_new = rng.poisson(params.lambda_poisson * state.K_AI)
    enqueue_new_ideas(
        queue=state.queue,
        n=num_new,
        t_arrival=t,
        mu_prior=state.mu_prior,
        tau_prior=params.tau_prior,
        sigma_noise=params.sigma_noise,
        rng=rng,
    )

    # 2) Screening throughput & accuracy ------------------------------------
    U_tot   = state.Uf  + params.xi1 * state.Unf  * (state.Hnf ** params.zeta_skill_exp)

    psi_eff = psi_efficiency(
        uf=state.Uf,
        unf=state.Unf,
        h_nf=state.Hnf,
        params=params,
        u_bar_mean=U_bar_others,
    )

    theta   = theta_accuracy(
        uf=state.Uf,
        unf=state.Unf,
        h_nf=state.Hnf,
        params=params,
    )


    # 3) Service the FIFO idea queue ----------------------------------------
    served, latency_stats = service_queue_fifo(queue=state.queue,
                                               capacity=psi_eff,
                                               t_now    = t,   
                                               rng=rng)

    # 4) Production ----------------------------------------------------------
    Y = output_ecb(
        alpha=state.alpha,
        labor_current=state.L_Y,
        x_values=state.x_values,
        psi_eff=psi_eff,
        theta_tot=theta,
        eta_clip=params.eta_clip,
    )

    # 5) Capital & skill updates --------------------------------------------
    state.Uf   *= (1.0 - params.delta_Uf)
    state.Unf  *= (1.0 - params.delta_Unf)
    state.Hnf   = (1.0 - params.delta_H) * state.Hnf  + params.mu_learning * psi_eff

    # 6) Return KPIs ---------------------------------------------------------
    return {
        "t":             t,
        "firm_id":       state.firm_id,
        "Y_new":         Y,
        "congestion_idx": U_bar_others, 
        "psi_eff":       psi_eff,
        "theta":         theta,
        "queue_len":     len(state.queue),
        "mean_latency":  latency_stats.mean if latency_stats.count else np.nan,
        "p95_latency":   latency_stats.p95  if latency_stats.count else np.nan,
    }
