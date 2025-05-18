# utils/financial_utils.py
# Cash-flow helpers used by the investment optimiser 
from typing import Mapping, Union

__all__ = ["available_cashflow", "price_index"]

Number = Union[int, float]


def available_cashflow(
    revenue: Number,
    costs:   Union[Number, Mapping[str, Number]],
    retention_ratio: float
) -> float:
    """
    Compute *retained* cash-flow available for new investments.

        cash_t = max(0, (revenue_t − total_costs_t)) × retention_ratio

    Parameters
    ----------
    revenue : float
        Gross period revenue *Y_t*.  Must be finite.
    costs : float | Mapping[str, float]
        Either a scalar (aggregate operating costs) or
        a mapping of cost categories → amounts.  Values must be finite.
    retention_ratio : float
        Fraction of net profit the firm reinvests.  Must lie in [0, 1].

    Returns
    -------
    float
        Non-negative cash amount earmarked for capital outlays.

    Raises
    ------
    ValueError
        If inputs are outside admissible domains.
    """
    # validation 
    if not (0.0 <= retention_ratio <= 1.0):
        raise ValueError(f"retention_ratio must be in [0,1], got {retention_ratio}")

    if isinstance(costs, Mapping):
        total_costs: float = sum(float(v) for v in costs.values())
    else:
        total_costs = float(costs)

    net_profit = float(revenue) - total_costs
    retained   = max(net_profit, 0.0) * retention_ratio
    return retained

def price_index(
    output_y: Number,
    *,
    price_ref: float,
    y_ref: float,
    elasticity: float,
    price_floor: float = 1e-6,
) -> float:
    """
    Inverse-demand price index used by *ecb_firm_step* to convert real
    output *Y* into nominal revenue.

        P_t = price_ref · (Y_t / y_ref)^(–ε)

    Parameters
    output_y : float
        Current real output Y_t (must be > 0).
    price_ref : float
        Benchmark price when output equals *y_ref*.
    y_ref : float
        Reference output level (scale parameter).  Must be > 0.
    elasticity : float
        Demand elasticity ε ≥ 0.  A value of 0 keeps price constant.
    price_floor : float, optional
        Hard lower bound to prevent negative or zero prices.

    Returns
    -------
    float
        The nominal price P_t ≥ price_floor.
    """
    if output_y <= 0 or y_ref <= 0 or price_ref <= 0:
        raise ValueError("output_y, y_ref and price_ref must all be > 0")
    if elasticity < 0:
        raise ValueError("elasticity must be ≥ 0")

    ratio  = output_y / y_ref
    price  = price_ref * (ratio ** (-elasticity)) if elasticity > 0 else price_ref
    return max(price, price_floor)
