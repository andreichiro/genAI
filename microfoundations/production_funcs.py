# microfoundations/production_funcs.py
# CES AI–Labour production block + evaluator-cost curve
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "CESParams",
    "ces_output",
    "marginal_products",
    "eval_cost_curve",
]

# 1.  Parameter bundle 

@dataclass(frozen=True, slots=True)
class CESParams:
    """
    Immutable container for the micro production parameters.

    α, ρ follow the usual CES notation;  A is total-factor productivity.

        Y   =  A · [ α · K_AI^ρ  +  (1−α) · L_eval^ρ ]^(1/ρ)
    """
    alpha: float = 0.35       # capital share
    rho:   float = -0.5       # elasticity of substitution  σ = 1/(1−ρ)
    A:     float = 1.0        # Hicks-neutral TFP

    def __post_init__(self):
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        # rho ∈ (−∞,1]  but exclude 1 (would reduce to perfect substitutes)
        if self.rho >= 1.0:
            raise ValueError("rho must be < 1")
        if self.A <= 0:
            raise ValueError("A must be positive")


# 2.  CES production function  

def ces_output(
    K_ai: float,
    L_eval: float,
    p: CESParams,
) -> float:
    """
    Constant-Elasticity-of-Substitution output Y.

    Parameters
    ----------
    K_ai : float
        AI capital (e.g. GPU compute, model capacity)  ≥ 0.
    L_eval : float
        Effective evaluator labour input  ≥ 0.
    p : CESParams
        Parameter bundle (α, ρ, A).

    Returns
    -------
    Y : float  ≥ 0
    """
    if min(K_ai, L_eval) < 0:
        raise ValueError("inputs must be non-negative")

    # handle Leontief (ρ → −∞) and Cobb–Douglas (ρ → 0) limits explicitly
    if p.rho == 0.0:                       # Cobb–Douglas
        return p.A * (K_ai ** p.alpha) * (L_eval ** (1.0 - p.alpha))
    if p.rho == -math.inf:                 # Leontief
        return p.A * min(
            K_ai / (p.alpha ** (1.0 / abs(p.rho))),
            L_eval / ((1.0 - p.alpha) ** (1.0 / abs(p.rho))),
        )

    term = p.alpha * (K_ai ** p.rho) + (1.0 - p.alpha) * (L_eval ** p.rho)
    return p.A * (term ** (1.0 / p.rho))


# 3.  Marginal products 

def marginal_products(
    K_ai: float,
    L_eval: float,
    p: CESParams,
) -> Tuple[float, float]:
    """
    ∂Y/∂K_ai  and  ∂Y/∂L_eval  under the CES technology.
    """
    Y = ces_output(K_ai, L_eval, p)
    if Y == 0.0:
        return 0.0, 0.0

    share_K = p.alpha * (K_ai ** p.rho) / (
        p.alpha * (K_ai ** p.rho) + (1.0 - p.alpha) * (L_eval ** p.rho)
    )
    share_L = 1.0 - share_K

    # elasticity σ = 1/(1−ρ)
    sigma = 1.0 / (1.0 - p.rho)

    MP_K = share_K * Y / K_ai if K_ai > 0 else 0.0
    MP_L = share_L * Y / L_eval if L_eval > 0 else 0.0

    # adjust for elasticity
    MP_K *= sigma
    MP_L *= sigma
    return MP_K, MP_L


# 4.  Evaluator cost curve  c_E(E)  

def eval_cost_curve(
    E: float,
    kappa: float = 0.10,
    phi: float = 1.20,
) -> float:
    """
    Unit cost of screening an additional idea when E ideas are already
    processed this period.  Convex form:

        c_E = (1 + κ · E)^φ

    • κ  controls how quickly congestion drives up costs.
    • φ  curvature (> 1 yields convexity;  φ=1 reduces to linear).

    The function is normalised so c_E=1 when E=0.

    Returns
    -------
    float  ≥ 1
    """
    if E < 0:
        raise ValueError("E must be ≥ 0")
    if kappa < 0 or phi < 0:
        raise ValueError("kappa and phi must be ≥ 0")
    return (1.0 + kappa * E) ** phi
