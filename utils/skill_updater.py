# utils/skill_updater.py

# Utility helper for human-capital (evaluator-skill) dynamics
# in the ECB evaluation-capacity pipeline.

import numpy as np
from typing import Union

__all__ = ["update_skill"]

def update_skill(
    H_prev: Union[float, np.ndarray],
    U_nf:   Union[float, np.ndarray],
    mu:     float,
    delta_H: float,
    psi_eff: Union[float, np.ndarray], 
) -> Union[float, np.ndarray]:
    """
    One-step learning-by-doing update for the non-fungible evaluator
    skill stock *H* (human capital):

        H_{t+1} = (1 − δ_H) · H_t  +  μ · Ψ_eff,t  

    In the baseline model, the learning term is linear in the current
    **non-fungible evaluation capital** *U_nf*.  More sophisticated
    concavities can be layered later via a strategy pattern without
    touching this core helper.

    Parameters
    ----------
    H_prev : float | np.ndarray
        Skill stock carried into the period.  Must be non-negative.
    psi_eff : float | np.ndarray
        Effective evaluation throughput in the current period. Must be ≥ 0.

    mu : float
        Learning-by-doing coefficient μ ≥ 0.
    delta_H : float
        Depreciation rate of human capital δ_H ∈ [0, 1).

    Returns
    -------
    float | np.ndarray
        Updated skill stock H_{t+1}.

    Raises
    ------
    ValueError
        If any input is out of the admissible domain.
    """
    # validation
    if mu < 0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    if not (0.0 <= delta_H < 1.0):
        raise ValueError(f"delta_H must be in [0,1), got {delta_H}")

    arr_H = np.asarray(H_prev, dtype=float)
    arr_ψ = np.asarray(psi_eff, dtype=float)
    if np.any(arr_H < 0):
        raise ValueError("H_prev must be non-negative")
    if np.any(arr_ψ < 0):
        raise ValueError("psi_eff must be non-negative")    # ← [NEW]        

    # update 
    retained  = (1.0 - delta_H) * arr_H
    acquired  = mu * arr_ψ                 # learning-by-doing term
    H_next    = retained + acquired
    return H_next
