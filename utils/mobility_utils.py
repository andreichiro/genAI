"""
mobility_utils.py

Handles cross-firm mobility of **non-fungible evaluators** (U_nf) that creates a strategic “talent-poaching” moat.

Flow rule:
ΔU_nf,i  =  ε · (w_i − w̄)
where
    • ε  = `mobility_elasticity`  (≥0, from Scenario/ECBParams/YAML)
    • w̄ = simple average wage across all firms that report a wage
The sum of all ΔU_nf is zero, so evaluator stock is conserved.

Public helper:
poach_evaluators(firm_states, wages, elasticity)
    Mutates the `Unf` field *in-place* on each FirmECBState.
"""
from __future__ import annotations

import logging
from typing import Dict, Sequence
from typing import TYPE_CHECKING 

if TYPE_CHECKING:                      
    from multifirm_runner import FirmECBState

def poach_evaluators(
    firm_states: Sequence[FirmECBState],
    wages: Dict[int, float],
    elasticity: float,
) -> None:
    """
    Redistribute evaluator capital between firms according to wage gaps.

    Parameters
    firm_states :
        List of live FirmECBState objects (mutated in-place).
    wages :
        Mapping  {firm_id → average evaluator wage}.  Firms that omit
        a wage are ignored in the flow this period.
    elasticity :
        Mobility elasticity ε ≥ 0.  Zero disables mobility.
    """
    if elasticity <= 0.0 or not wages:          # nothing to do
        return

    w_bar = sum(wages.values()) / len(wages)    # sector-wide mean wage
    # gap_i = w_i − w̄
    gaps = {fid: wages[fid] - w_bar for fid in wages}

    # apply ΔU_nf = ε · gap
    for st in firm_states:
        if st.firm_id not in gaps:
            continue            # firm reported no wage this tick
        delta = elasticity * gaps[st.firm_id]
        st.Unf = max(st.Unf + delta, 0.0)   # clamp at zero – no negatives

    logging.debug(
        "poach_evaluators ▸ ε=%.3f  w̄=%.3f  total_net_flow=%.3f",
        elasticity,
        w_bar,
        sum(elasticity * g for g in gaps.values()),
    )
