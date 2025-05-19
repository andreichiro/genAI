# utils/triage_params.py
"""
TriageParams

Container for all per-scenario knobs that govern **Bayesian triage**:

    • σ            – signal noise std-dev
    • τ₀           – prior std-dev
    • λ_explore    – creativity / skepticism weight
    • threshold_rule
    • threshold_value

The dataclass is *frozen* so scenarios stay immutable after construction.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TriageParams:
    # noise & priors
    sigma_noise:   float = 1.0       # σ  > 0
    tau_prior:      float = 1.0       # τ₀ > 0

    # exploration
    lambda_explore: float = 0.0       # λ  (can be <0 for conservative)

    # thresholding
    threshold_rule:  str   = "percentile"   # "percentile" | "absolute"
    threshold_value: float = 50.0           # pct in [0,100]  or absolute cut-off

    def __post_init__(self) -> None:        # lightweight validation
        if self.sigma_noise <= 0:                       
            raise ValueError("sigma_noise must be > 0")   
        if self.tau_prior <= 0:
            raise ValueError("tau_prior must be > 0")
        if self.threshold_rule not in {"percentile", "absolute"}:
            raise ValueError("threshold_rule must be 'percentile' or 'absolute'")
        if self.threshold_rule == "percentile" and not (0.0 <= self.threshold_value <= 100.0):
            raise ValueError("percentile threshold_value must be in [0,100]")

    @property
    def sigma_signal(self) -> float:                           # pragma: no cover
        import warnings
        warnings.warn(
            "TriageParams.sigma_signal is deprecated; "
            "use sigma_noise instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sigma_noise
