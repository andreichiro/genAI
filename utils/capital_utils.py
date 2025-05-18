# utils/capital_utils.py
# Utility helpers for evaluation-capital dynamics in the ECB pipeline.

import numpy as np
from typing import Union

__all__ = ["phi_capital", "eval_updater"]

def phi_capital(I: Union[float, np.ndarray], phi: float) -> Union[float, np.ndarray]:
    """
    ϕ-concave installation function used for both fungible (Uf) and
    non-fungible (Unf) evaluation capital accumulation.

        ϕ(I) = I**φ       with   0 < φ ≤ 1            (φ < 1 ⇒ diminishing returns)

    Parameters
    ----------
    I : float | np.ndarray
        Gross investment during the period. Must be non-negative.
    phi : float
        Concavity parameter φ in (0, 1].  A value of 1 reduces to the
        linear ‘one-for-one’ installation case.

    Returns
    -------
    float | np.ndarray
        Effective capital installed this period.

    Raises
    ------
    ValueError
        If inputs are out of domain.
    """
    # validation 
    if phi <= 0 or phi > 1:
        raise ValueError(f"phi must be in (0,1], got {phi}")
    if np.any(np.asarray(I) < 0):
        raise ValueError("Investment I must be non-negative")

    # computation
    return np.power(I, phi)


def eval_updater(U_prev: Union[float, np.ndarray],
                 I: Union[float, np.ndarray],
                 delta: float,
                 phi: float) -> Union[float, np.ndarray]:
    """
    One-step update for *either* evaluation-capital stock (fungible or
    non-fungible):

        U_{t+1} = (1 − δ) · U_t  +  ϕ(I_t)

    Parameters
    ----------
    U_prev : float | np.ndarray
        Stock carried into the period.
    I : float | np.ndarray
        Investment during the period.
    delta : float
        Depreciation rate δ ∈ [0, 1).
    phi : float
        Concavity parameter passed through to `phi_capital`.

    Returns
    -------
    float | np.ndarray
        Updated stock U_{t+1}.

    Raises
    ------
    ValueError
        If inputs are out of domain.
    """
    # --- validation ---------------------------------------------------
    if not (0.0 <= delta < 1.0):
        raise ValueError(f"delta must be in [0,1), got {delta}")
    if np.any(np.asarray(U_prev) < 0):
        raise ValueError("U_prev must be non-negative")

    # --- update -------------------------------------------------------
    retained = (1.0 - delta) * U_prev
    installed = phi_capital(I, phi)
    return retained + installed
