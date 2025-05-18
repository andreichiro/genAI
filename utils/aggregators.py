# utils/aggregators.py

# Classical & CES aggregators retained from the legacy Romer codebase.

from __future__ import annotations
from typing import Optional

__all__ = ["aggregator_ces"]

# ────────────────────────────────────────────────────────────────────────────
def aggregator_ces(
    *,
    capital_current: float,
    labor_current: float,
    beta: float,
    tfp_val: float,
    rho: float,
    synergy: float = 0.0,
    intangible: float = 0.0,
    **kwargs,
) -> float:
    """
    Hicks-neutral CES aggregator:

        Y = A · [ β·K^ρ + (1-β)·L^ρ ]^(1/ρ)

    where σ = 1 / (1-ρ) is the elasticity of substitution.
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0,1]; got {beta}")
    # synergy / intangible hooks (kept for backward compatibility)
    tfp_val = tfp_val * (1.0 + synergy + intangible)

    K_term = beta * (capital_current ** rho)
    L_term = (1.0 - beta) * (labor_current ** rho)
    inside = K_term + L_term
    if inside <= 0:
        return 0.0

    return tfp_val * (inside ** (1.0 / rho))
