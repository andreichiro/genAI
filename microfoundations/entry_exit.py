# microfoundations/entry_exit.py                                                             
# hook for endogenous industry turnover.                           
# • Exit: firms with persistently negative return on assets are wound down. 
# • Entry: a Poisson flow of “green-field” entrants reacts to demand growth.
# The module is intentionally *stateless*: the mean-field runner (or any    
# other caller) owns firm lists and passes in the minimal history buffer    
# required to take decisions.                                               

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Sequence

import numpy as np
from numpy.random import Generator

__all__ = [
    "check_exit",
    "plan_entries",
]

# Exit rule 
def check_exit(
    roa_hist: Deque[float],
    streak: int = 3,
    roa_thresh: float = -0.05,
) -> bool:
    """Return **True** if the firm exits this period.

    Exit triggers when the last ``streak`` observations of Return-on-Assets
    have all fallen below ``roa_thresh``.
    The caller is responsible for clipping/cleaning NaNs before passing the
    deque.
    """
    if len(roa_hist) < streak:                                              
        return False                                                        
    recent = list(roa_hist)[-streak:]                                       
    return all(x < roa_thresh for x in recent)                              
                                                                            
                                                                            
# entry rule 
def plan_entries(
    demand_growth: float,
    coeff: float = 0.2,
    rng: Generator | None = None,
) -> int:
    """
    Number of entrants E_t ~ Poisson( λ ),  where
        λ = max(0, coeff · demand_growth).

    ``demand_growth`` should be the *sector-wide* Δln Q between periods.
    A zero or negative growth yields λ = 0 ⇒ no entry.
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = max(0.0, coeff * demand_growth)                                   
    return int(rng.poisson(lam))                                            
