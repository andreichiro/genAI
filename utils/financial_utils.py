# utils/financial_utils.py
# Cash-flow helpers used by the investment optimiser 
from typing import Mapping, Union

__all__ = ["available_cashflow"]

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
