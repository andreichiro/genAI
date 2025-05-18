# utils/optimizer_utils.py

# Myopic investment allocation for GPU, fungible, and non-fungible

from typing import Tuple
from utils.ecb_params import ECBParams       # access wage_ref benchmark

__all__ = ["solve_investments", "enrol_budget_fraction"] 

def solve_investments(
    cash: float,
    mp_gpu: float,          # ∂Π/∂K_AI  – marginal product of AI-generation capital
    mp_f: float,            # ∂Π/∂U_f   – marginal product of fungible eval capital
    mp_nf: float,           # ∂Π/∂U_nf  – marginal product of non-fungible eval capital
    mp_skill: float = 0.0,          
    min_allocation: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Split available cash across three investment buckets proportionally
    to (clipped) *current* marginal products – a myopic rule that ignores
    future dynamics but is fast and transparent.

    Parameters
    ----------
    cash : float
        Non-negative budget to allocate this period.
    mp_gpu : float
        Marginal product of an additional unit of K_AI (GPU / model capacity).
    mp_f : float
        Marginal product of an additional unit of fungible evaluation capital.
    mp_nf : float
        Marginal product of an additional unit of non-fungible evaluation capital.
    min_allocation : float, optional
        Hard minimum each bucket receives, even if its marginal product is
        zero.  Defaults to 0.  Must satisfy 3·min_allocation ≤ cash.

    Returns
    Tuple[float, float, float, float]
        (I_gpu, I_f, I_nf, I_train) – cash allocated to GPU capacity,
+        fungible capital, non-fungible capital, and evaluator-training
+        respectively.
    
    Raises
    ------
    ValueError
        If inputs are invalid or constraints violated.
    """
    if cash < 0:
        raise ValueError("cash must be non-negative")
    if min_allocation < 0:
        raise ValueError("min_allocation must be ≥ 0")
    if 4 * min_allocation > cash:    
        raise ValueError("min_allocation too large for given cash pool")

    #  clip & normalise marginal products 
    weights = [
            max(mp_gpu,   0.0),
            max(mp_f,     0.0),
            max(mp_nf,    0.0),
            max(mp_skill, 0.0)           
        ]

    residual_cash = cash - 4 * min_allocation

    if residual_cash <= 0 or sum(weights) == 0:
         # Either no extra money or all MPs zero ⇒ equal split of minima (4 buckets)
        return (min_allocation, min_allocation, min_allocation)

    weight_sum = float(sum(weights))
    shares = [w / weight_sum for w in weights]

    allocations = [min_allocation + residual_cash * s for s in shares]
    I_gpu, I_f, I_nf, I_train = allocations

    return I_gpu, I_f, I_nf, I_train 

def enrol_budget_fraction(avg_wage: float, p: ECBParams) -> float:
    """
    Simple rule: when the average evaluator wage exceeds a reference
    level (``p.wage_ref``) we devote up to 25 % of the evaluation-capital
    budget to fund new trainees.

        frac = min(0.25, max(0, avg_wage / p.wage_ref - 1))

    Returns a number in [0, 0.25].  If ``p.wage_ref`` ≤ 0 the function
    falls back to 0.
    """
    if p.wage_ref <= 0:
        return 0.0
    excess = avg_wage / p.wage_ref - 1.0
    return min(0.25, max(0.0, excess))
