# microfoundations/demand.py                                                                
# inverse-demand helper                                           
# Smooth CES-style curve usable by ecb_firm_step when pricing output.      

from __future__ import annotations
import math

__all__ = ["price_at_quantity", "elasticity_at_quantity"]

# Inverse-demand curve P(Q)
def price_at_quantity(
    Q: float,
    P_max: float = 10.0,     # choke-price at zero supply
    Q_half: float = 100.0,   # quantity where price falls to ½·P_max
    eta: float = 1.5,        # curvature (>0)
) -> float:
    """
    P(Q) = P_max / (1 + (Q / Q_half)^η)

    Guarantees P(Q) ∈ (0, P_max] and dP/dQ < 0 for Q > 0.
    """
    if Q < 0:
        raise ValueError("Q must be non-negative")
    denom = 1.0 + (Q / Q_half) ** eta
    return P_max / denom

# ‣ Point elasticity ε(Q)
def elasticity_at_quantity(
    Q: float,
    P_max: float = 10.0,
    Q_half: float = 100.0,
    eta: float = 1.5,
) -> float:
    """
    ε(Q) = |dP/dQ| · Q / P(Q)  for the curve above.
    Returns ∞ at Q = 0 (vertical demand) and declines thereafter.
    """
    if Q == 0:
        return math.inf
    P = price_at_quantity(Q, P_max, Q_half, eta)
    ratio = Q / Q_half
    # dP/dQ exactly derived from the closed-form P(Q)
    dP_dQ = -(eta * P_max / Q_half) * (ratio ** (eta - 1)) / (1.0 + ratio ** eta) ** 2
    return abs(dP_dQ * Q / P)

