# utils/screening_utils.py

import logging
import numpy as np                    
from numpy.random import Generator
from typing import Final
from typing import Tuple, Sequence
from utils.ecb_params import ECBParams      
from utils.skill_updater import update_skill

import warnings                                                      
from utils.triage_utils import (                                   
    bayes_posterior as _bayes_post,                              
    triage_score   as _triage_score,                             
)                  

def bayes_update(*args, **kwargs):                                        
    warnings.warn(                                                            
        "screening_utils.bayes_update is deprecated; "
        "use utils.triage_utils.bayes_posterior",
        DeprecationWarning,
        stacklevel=2,
    )
    return _bayes_post(*args, **kwargs)                                     

def triage_score(*args, **kwargs):                                            
    warnings.warn(                                                            
        "screening_utils.triage_score is deprecated; "
        "use utils.triage_utils.triage_score",
        DeprecationWarning,
        stacklevel=2,
    )
    return _triage_score(*args, **kwargs)                                     


def gen_ideas(
    kai: float,
    params: ECBParams,
    rng: Generator
) -> int:
    """
    Draws the number of new candidate ideas in the current period.

    N ~ Poisson(λ · K_AI)

    Parameters
    ----------
    kai : float
        The firm’s AI-generation capital \(K_{AI,t}\) (≥ 0).
    params : ECBParams
        Container of model scalars; must provide `lambda_poisson`.
    rng : numpy.random.Generator
        A NumPy RNG passed down to keep simulation reproducible
        under joblib / multiprocessing.

    Returns
    -------
    int
        Realisation of the Poisson random variable. Guaranteed ≥ 0.
    """
    if kai < 0:
        raise ValueError(f"K_AI must be ≥ 0, got {kai}")
    lam: Final = params.lambda_poisson * kai
    if lam < 0:
        # should never happen, but we guard anyway
        raise ValueError(f"Poisson λ became negative ({lam})")
    return int(rng.poisson(lam=lam))


def theta_total(
    uf: float,
    unf: float,
    h_nf: float,
    params: ECBParams,
    accuracy_t: float = 1.0,
) -> float:
    """
    Total screening accuracy θ_t ∈ [0,1].

        U_tot  = Uf + ξ₁ · Unf · H_nf^ζ
        θ_cap  = 1 – exp(–ξ_success · U_tot)
        θ_skill= 1 – exp(–χ_skill  · H_nf)
        θ_tot  = θ_cap · θ_skill · accuracy_t   (clamped to 1)

    Parameters
    ----------
    uf, unf : float
        Fungible / non-fungible evaluation capital stocks (≥ 0).
    h_nf : float
        Human evaluator skill stock (≥ 0).
    params : ECBParams
        Model parameters (ξ₁, ζ_skill_exp, ξ_success, χ_skill).
    accuracy_t : float, optional
        Period-specific residual accuracy multiplier (defaults 1).

    Returns
    -------
    float
        θ_tot clamped to ≤ 1.
    """
    if min(uf, unf, h_nf) < 0:
        raise ValueError("Capital and skill inputs must be ≥ 0")

    u_tot = uf + params.xi1 * unf * (h_nf ** params.zeta_skill_exp)

    theta_cap   = 1.0 - np.exp(-params.xi_success * u_tot)
    theta_skill = 1.0 - np.exp(-params.chi_skill  * h_nf)
    theta       = theta_cap * theta_skill * accuracy_t

    if theta > 1.0 + 1e-12:          # numeric margin
        logging.debug("θ_total %.4f > 1 — clamped to 1", theta)
    return min(theta, 1.0)

def _psi_inv_u(u_tot: float, p: ECBParams) -> float:
    """
    Inverted-U evaluation-capacity curve (over-evaluation drag).

        Ψ_raw = ψ0 + (ψ_max–ψ0) · ( (u_tot / U⋆) · exp(1 − u_tot/U⋆) )

    • Peaks at U_tot = U⋆, then falls → captures bureaucratic bloat.
    • Returns ψ0 when U_tot → 0, guaranteeing continuity.
    """
    mid   = max(p.U_star, 1e-9)                     # avoid divide-by-zero
    width = p.psi_max - p.psi0
    return p.psi0 + width * ((u_tot / mid) * np.exp(1.0 - u_tot / mid))

def screening_capacity(
    uf: float,
    unf: float,
    h_nf: float,
    params: ECBParams,
    u_bar_mean: float,
) -> float:
    """
    Effective evaluation throughput Ψ_eff given capital, skill and congestion.

        U_tot = Uf + ξ₁·Unf·H_nf^ζ
        Ψ_raw = ψ0 + (ψ_max – ψ0) / (1 + exp(–κ·(U_tot – U_star)))
        Ψ_eff = Ψ_raw / (1 + η · Ū_{-i})

    Parameters
    ----------
    uf, unf, h_nf : float
        Capital & skill stocks (≥ 0).
    params : ECBParams
        Must include ψ0, ψ_max, kappa, U_star, eta_congestion.
    u_bar_mean : float
        Congestion term: mean evaluation capital of rival firms.

    Returns
    -------
    float
        Throughput rate Ψ_eff ≥ 0.
    """
    if min(uf, unf, h_nf, u_bar_mean) < 0:
        raise ValueError("Inputs must be ≥ 0")

    u_tot = uf + params.xi1 * unf * (h_nf ** params.zeta_skill_exp)

    if params.psi_shape == "inv_u":
        psi_raw = _psi_inv_u(u_tot, params)
    else:  # default “logistic”
         z = -params.kappa * (u_tot - params.U_star)
         z_clipped = np.clip(z, -700.0, 700.0)     # exp() safe range
         denom = 1.0 + np.exp(z_clipped)
         psi_raw = params.psi0 + (params.psi_max - params.psi0) / denom

    return psi_raw / (1.0 + params.eta_congestion * u_bar_mean)

# ‣ REQ-TICK adaptive threshold helper (percentile or absolute)
def compute_threshold(
    scores: Sequence[float],
    rule: str = "percentile",
    value: float = 0.0,
) -> float:
    warnings.warn(
        "screening_utils.compute_threshold is deprecated; "
        "use utils.triage_utils.apply_threshold for Boolean masks "
        "or migrate to the Scenario YAML threshold rules.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not scores:
        return float("inf")          # empty batch → nothing selected


psi_efficiency = screening_capacity     # alias for clarity
theta_accuracy = theta_total           # alias for clarity

__all__ = [
    "psi_efficiency",
    "theta_accuracy",
    "bayes_update",
    "triage_score",
    "compute_threshold",
    "update_skill",                              
]