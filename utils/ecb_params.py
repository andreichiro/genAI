# utils/ecb_params.py
"""
ECBParams

Central container for every scalar that governs the **E**valuation-

Default values replicate the `defaults:`
section we inserted in *scenarios.yaml*, so legacy simulations that pass `None` continue to behave exactly as before.

All attributes are fully type-annotated to enable static checking.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class ECBParams:
    # Screening-capacity block 
    xi1:              float = 1.0     # weighting on Unf · H^ζ
    zeta_skill_exp:   float = 1.0     # ζ exponent on H_nf
    xi_success:       float = 0.15    # slope in θ_cap
    chi_skill:        float = 0.10    # slope in θ_skill

    psi0:             float = 0.0     # min throughput
    psi_max:          float = 10.0    # max throughput
    U_star:           float = 5.0     # midpoint of logistic ψ(U_tot)
    kappa:            float = 1.0     # logistic steepness
    psi_shape:        str = "logistic"  # "logistic" or "inv_u"
    eta_congestion:   float = 0.05    # Ψ_eff divisor coefficient
    mobility_elasticity: float = 0.0  # ε ≥0 in evaluator poaching flow
    wage_ref:            float = 1.0  # benchmark wage triggering training

    #  Idea-arrival 
    lambda_poisson:   float = 0.10    # Poisson λ per unit K_AI
    tau_prior:        float = 1.0     # Prior std-dev τ of true idea value      ◄ NEW
    sigma_noise:      float = 1.0     # Signal noise std-dev σ                  ◄ NEW

    shared_pool:        bool  = False     
    tau_spillover: float       = 0.0 
    # Clip factor in production 
    eta_clip:         float = 1.0     # scaling in min(x, η·Ψ_eff)
    eta_decay:        float = 0.0     # κ_d in e^(−κ_d·lag); 0 ⇒ disabled
    v0_mu:            float = 1.0     # mean of log-normal base idea value
    v0_sigma:         float = 0.20    # σ  of log-normal base idea value

    # Capital & skill accumulation 
    phi_concavity:    float = 0.75    # ϕ(I)=I^φ concavity
    mu_learning:      float = 0.05    # H_nf learning-by-doing rate

    # Knowledge success probability 
    q_success:        float = 0.30

    # Cost-curve curvatures (optimiser) 
    curvature_gpu:    float = 2.0
    curvature_f:      float = 2.0
    curvature_nf:     float = 2.0
    curvature_skill:  float = 2.0

    # Depreciation
    delta_Uf:         float = 0.05
    delta_Unf:        float = 0.05
    delta_H:          float = 0.02

    # education pipeline 
    education_lag:  int   = 0        # periods from enrolment ➜ graduation
    enroll_rate:    float = 0.0      # share of trainees enrolled each tick
    retire_rate:    float = 0.0      # attrition among trainees
    enroll_const:   float = 0.0      # constant enrolment
    initial_trainees: float = 0.0    # initial trainee stock
    
    # The dataclass purposefully contains **no behaviour**; all formulas
    # live in domain modules such as `screening_utils.py`.
