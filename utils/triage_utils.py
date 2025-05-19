# utils/triage_utils.py
# ────────────────────────────────────────────────────────────────────────────
# Bayesian update + triage-score helpers for the Evaluator-Centric model.
#
# The formulas replicate Section 2 of the theory:
#
#   v  ~  N( μ_prior , τ² )
#   s  =  v + ε ,     ε ~ N( 0 , σ² )
#
# → posterior moments:
#     μ_post = ( τ² / (τ² + σ²) ) · s  +  ( σ² / (τ² + σ²) ) · μ_prior
#     τ²_post = ( τ² · σ² ) / ( τ² + σ² )
#
# Triaging weight:
#     T = μ_post  +  λ · τ²_post
#
# The helper functions are intentionally stateless so they can be JIT-ed or
# vectorised externally without refactoring this module.

from __future__ import annotations

import numpy as np
from typing import Tuple

__all__ = [
    "bayes_posterior",
    "triage_score",
    "apply_threshold",    
]

def bayes_posterior(
    *,
    signal: float,
    mu_prior: float,
    tau_prior: float,
    sigma_noise: float,
) -> Tuple[float, float]:
    """
    Compute posterior mean and variance given a single noisy signal.

    Parameters
    ----------
    signal : float
        Observed signal s = v + ε.
    mu_prior : float
        Prior mean μ_prior.
    tau_prior : float
        Prior standard deviation τ (> 0).
    sigma_noise : float
        Noise standard deviation σ (> 0).

    Returns
    -------
    (μ_post, τ²_post) : tuple[float, float]
        Posterior mean and variance.
    """
    if tau_prior <= 0.0:
        raise ValueError("tau_prior must be > 0")
    if sigma_noise <= 0.0:
        raise ValueError("sigma_noise must be > 0")

    tau2   = tau_prior ** 2
    sigma2 = sigma_noise ** 2
    denom  = tau2 + sigma2

    mu_post   = (tau2 / denom) * signal + (sigma2 / denom) * mu_prior
    var_post  = (tau2 * sigma2) / denom
    return float(mu_post), float(var_post)

def apply_threshold(                               
    scores: np.ndarray,
    *,
    rule: str,
    value: float,
) -> np.ndarray:
    """
    Vectorised triage gate.

    Parameters
    ----------
    scores : np.ndarray
        Array of triage scores  *T*  (shape: N,).
    rule : {"percentile", "absolute"}
        *percentile* – keep  scores >= np.percentile(scores, value)  
        *absolute*   – keep  scores >= value
    value : float
        Percentile (0–100) or absolute cut-off, depending on *rule*.

    Returns
    -------
    np.ndarray[bool]
        Boolean mask of accepted ideas, same shape as *scores*.
    """
    if not isinstance(scores, np.ndarray):
        scores = np.asarray(scores, dtype=float)


    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D array")

    if rule == "percentile":
        if not (0.0 <= value <= 100.0):
            raise ValueError("percentile must be in [0, 100]")
        cutoff = float(np.percentile(scores, value))
    elif rule == "absolute":
        cutoff = float(value)
    else:
        raise ValueError(f"unknown threshold_rule '{rule}'")

    return scores >= cutoff

def triage_score(
    *,
    mu_post: float,
    var_post: float,
    lambda_weight: float,
) -> float:
    """
    Compute the triage score   T = μ_post + λ · var_post.

    λ > 0  ⇒ exploratory weight on uncertainty  
    λ < 0  ⇒ sceptical / conservative
    """
    return float(mu_post + lambda_weight * var_post)
