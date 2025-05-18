# utils/aggregator_ecb.py
# ────────────────────────────────────────────────────────────────────────────
# Evaluation-constrained production aggregator for the ECB refactor
#
# Implements:
#     Y  =  tfp_val · L^{1-α} · [ Σ_i  min( x_i ,  η·Ψ_eff ) ]^{α} · θ_tot^{γ}
#
# Where
#   • L          = final-goods labour in period t
#   • x_i        = intermediate-good usage for variety i   (len = A_t)
#   • Ψ_eff      = effective screening throughput
#   • θ_tot      = total screening accuracy in [0,1]
#   • η          = clip factor (dataclass ECBParams.eta_clip)
#   • γ          = exponent capturing complementarity of accuracy (γ≥0)
#   • tfp_val    = optional multiplicative TFP term (defaults 1)
#

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np


def output_ecb(
    *,
    alpha: float,
    labor_current: float,
    x_values: Sequence[float],
    psi_eff: float | None = None,
    theta_tot: float | None = None,
    eta_clip: float = 1.0,
    gamma_acc: float = 1.0,
    tfp_val: float = 1.0,
    ideas_batch: Sequence[tuple[int, int, float, float, float]] | None = None,
    eta_decay: float = 0.0,
    return_loss: bool = False,          # if True → return (Y, lost_value)
    **kwargs: Any,
) -> float:
    """
    Evaluation-constrained Dixit–Stiglitz aggregator.

    Parameters
    ----------
    alpha : float
        Elasticity exponent in (0,1).
    labor_current : float
        Final-goods labour \(L_t\) (≥ 0).
    x_values : sequence[float]
        Intermediate-goods usage \(x_i\) (all ≥0).
    psi_eff : float, optional
        Effective evaluation throughput \(Ψ_{eff}\).  
        If *None*, treated as ∞ ⇒ no bottleneck (legacy behaviour).
    theta_tot : float, optional
        Screening accuracy \(θ_{tot}\in[0,1]\).  
        If *None*, treated as 1.
    eta_clip : float, default 1.0
        Scaling factor η in the min-clip term.
    gamma_acc : float, default 1.0
        Exponent γ on θ_tot.
    tfp_val : float, default 1.0
        Hicks-neutral TFP multiplier (synergy/intangible already folded in).

    Returns
    -------
    float
        Period output \(Y_t\).

    Raises
    ------
    ValueError
        If inputs are out of admissible ranges.
    """
    # ------------------------------------------------------------------ 1 : validate scalars
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    if labor_current < 0.0:
        raise ValueError(f"labor_current cannot be negative; got {labor_current}")
    if eta_clip <= 0.0:
        raise ValueError(f"eta_clip must be >0; got {eta_clip}")
    if gamma_acc < 0.0:
        raise ValueError(f"gamma_acc must be ≥0; got {gamma_acc}")
    if psi_eff is not None and psi_eff < 0.0:
        raise ValueError(f"psi_eff must be ≥0 or None; got {psi_eff}")
    if theta_tot is not None and not (0.0 <= theta_tot <= 1.0 + 1e-12):
        raise ValueError(f"theta_tot must be in [0,1]; got {theta_tot}")

    if eta_decay < 0.0:
        raise ValueError(f"eta_decay must be ≥0; got {eta_decay}")
    # ------------------------------------------------------------------ 2 : canonicalise optionals
    psi_eff_val = float("inf") if psi_eff is None else float(psi_eff)
    theta_val   = 1.0           if theta_tot is None else float(theta_tot)

    x_arr = np.asarray(x_values, dtype=float)
    if (x_arr < 0).any():
        raise ValueError("x_values contain negative entries")

    # ------------------------------------------------------------------ 3 : core formula
    clipped_sum = np.minimum(x_arr, eta_clip * psi_eff_val).sum()
    if clipped_sum < 0:
        # should be impossible barring NaN, but guard anyway
        raise ValueError("sum of clipped x_i became negative – check inputs")

    labor_term = labor_current ** (1.0 - alpha)
    goods_term = clipped_sum ** alpha
    acc_term   = theta_val ** gamma_acc

    Y = tfp_val * labor_term * goods_term * acc_term

    lost_value = 0.0
    if ideas_batch and eta_decay > 0.0:
        tot_nominal = 0.0
        realised    = 0.0
        for t_arr, t_eval, _mu, _var, v0 in ideas_batch:
            lag        = max(t_eval - t_arr, 0)
            discount   = np.exp(-eta_decay * lag)
            tot_nominal += v0
            realised    += v0 * discount
        lost_value = max(tot_nominal - realised, 0.0)


    logging.debug(
        "output_ecb: L=%.3f, Σmin(x,ηψ)=%.3f, θ=%.3f → Y=%.3f, lost=%.3f",
        labor_current, clipped_sum, theta_val, Y, lost_value,
    )
    Y_safe = float(max(Y, 0.0))          # explicit cast & non-negative safeguard
    if return_loss:
        return (Y_safe, float(lost_value))
    return Y_safe

__all__ = ["output_ecb"]
