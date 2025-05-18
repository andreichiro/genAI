# microfoundations/spillovers.py
# Cross-firm knowledge spill-over helper  

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["knowledge_spillover"]

def knowledge_spillover(
    U_nf_vec: Sequence[float],
    tau: float,
) -> float:
    """
    Aggregate externality Ω(U_nf) that raises each firm’s idea quality
    prior mean μ₀ through inter-firm knowledge spill-overs:

        Ω = τ · ȲU_nf          with   ȲU_nf = mean(U_nf_vec)

    The caller adds Ω to its own μ_prior before sampling new ideas.

    Parameters
    ----------
    U_nf_vec : sequence[float]
        Current non-fungible evaluator capital of *all* firms.
    tau : float ∈ [0,1]
        Spill-over intensity.  τ=0 disables the channel.

    Returns
    -------
    Ω : float ≥ 0
    """
    if tau <= 0.0 or len(U_nf_vec) == 0:
        return 0.0
    if tau > 1.0:
        raise ValueError("tau must be ≤ 1")

    mean_Unf = float(np.mean(U_nf_vec))
    return tau * mean_Unf
