# scenarios.py  
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any, Final      
from itertools import product
from collections import OrderedDict

import numpy as np
import yaml

from utils.ecb_params    import ECBParams
from utils.triage_params import TriageParams         

ARCHETYPES: Final = {
    "top",
    "intermediate",
    "bottom",
   "latency_decay",          #opportunity-cost latency scenarios
}

# data-class consumed by compute_y_romer & sim_runner
@dataclass(frozen=True, slots=True)
class ScenarioCfg:
    id: str

    # global scalars
    num_periods: int
    alpha: float
    dt_integration: float
    growth_explosion_threshold: float
    # model callables – **never None** 
    labor_func: Callable[[int], float]
    total_labor_func: Callable[[int], float]
    share_for_rd_func: Callable[[int], float]
    capital_func: Callable[[int], float]
    synergy_func: Callable[[int], float]
    intangible_func: Callable[[int], float]
    x_values_updater: Callable[
        [int, np.ndarray, float, float, float, float], np.ndarray
    ]
    # forward-compat hook 
    intangible_invest_func: Callable[[int], float]

    intangible_kappa: float
    intangible_Ubar: float
    intangible_epsilon: float
    skill_interaction_func: Callable[[int, float, float], float]
    skill_interaction_on:    bool
    ecb_params:    ECBParams
    triage_params: TriageParams

    test_label: str = ""
    hypothesis: str = ""
    tags: tuple[str, ...] = ()

@dataclass(slots=True, frozen=True)
class ScenarioECB:
    """
    Minimal configuration object consumed by `multifirm_runner.run_ecb`.
    """
    id:            str

    num_periods:   int
    firms_init:    List[Dict[str, Any]]        # per-firm starting stocks
    ecb_params:    ECBParams
    triage_params: TriageParams

    test_label:    str = ""
    hypothesis:    str = ""
    tags:          tuple[str, ...] = ()
    spillover_intensity: float = 0.0
    
# public API 
def load_scenarios(yaml_path: str | Path = "scenarios.yaml") -> List[ScenarioCfg]:
    """
    Parse `scenarios.yaml` and return a fully-typed list of ScenarioCfg
    ready to be fed into `compute_y_romer`.  All numeric knobs live in YAML;
    no magic numbers remain in Python.
    """
    data: Dict[str, Any] = yaml.safe_load(Path(yaml_path).read_text())
    defaults = data["defaults"]
    return [_build_scenario(row, defaults) for row in data["scenarios"]]

def load_scenarios_ecb(
    yaml_path: str | Path = "scenarios.yaml",
    *,
    grid: bool | None = None,          
) -> List[ScenarioECB]:
    """
    Parse the ECB-style section of `scenarios.yaml` and return a list of
    ScenarioECB objects. Only scenarios marked with engine: ecb are processed.
    """
    raw = yaml.safe_load(Path(yaml_path).read_text())

    defaults = raw.get("defaults", {})
    ecb_defaults = defaults.get("ecb_params", {})
    triage_defaults = defaults.get("triage_params", {})

    out: List[ScenarioECB] = []
    for row in raw["scenarios"]:
        # Only process scenarios explicitly marked as ECB
        if row.get("engine") == "ecb":
            # ECB scenarios must have firms_init
            if "firms_init" not in row:
                raise ValueError(f"ECB scenario '{row['id']}' is missing required 'firms_init' configuration")

            ecb_cfg = {**ecb_defaults, **row.get("ecb", {})}
            tri_cfg = {**triage_defaults, **row.get("triage", {})}

            scenario = ScenarioECB(
                id=row["id"],
                num_periods=row.get("num_periods", defaults["num_periods"]),
                firms_init=row["firms_init"],
                ecb_params=ECBParams(**ecb_cfg),
                triage_params=TriageParams(**tri_cfg),
                test_label=row.get("test_label", ""),
                hypothesis=row.get("hypothesis", ""),
                spillover_intensity=ecb_cfg.get("tau_spillover", 0.0), 
            )

            out.append(scenario)

    for auto in generate_matrix(raw["defaults"]):
        ecb_cfg = {**ecb_defaults, **auto.get("ecb", {})}
        tri_cfg = {**triage_defaults, **auto.get("triage", {})}
        out.append(
            ScenarioECB(
                id=auto["id"],
                test_label=auto["test_label"],
                hypothesis=auto["hypothesis"],
                num_periods=auto["num_periods"],
                firms_init=auto["firms_init"],
                spillover_intensity=ecb_cfg.get("tau_spillover", 0.0),
                ecb_params=ECBParams(**_subset_kwargs(ECBParams, ecb_cfg)),
                triage_params=TriageParams(**_subset_kwargs(TriageParams, tri_cfg)),
            )
        )

    return out

def generate_matrix(defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand the `defaults['matrix']` Cartesian grid into a list of raw-scenario
    dictionaries that mimic the manual YAML records.  Symbolic axis levels are
    translated through `defaults['lookup_tables']`.
    """
    if "matrix" not in defaults:          # nothing to do
        return []

    axes = OrderedDict(defaults["matrix"])
    keys, values = axes.keys(), axes.values()

    lut = defaults["lookup_tables"]
    scenarios: List[Dict[str, Any]] = []

    for combo in product(*values):
        spec = dict(zip(keys, combo))

        invest_block   = lut["invest_patterns"][spec["r_and_d_regime"]]
        shock_drop_pct = lut["shock_magnitude"][spec["shock_magnitude"]]
        eta_decay      = lut["latency_level"][spec["latency_level"]]
        mobility_eps   = lut["mobility"][spec["mobility"]]
        spill_int      = lut["spillover"][spec["spillover"]]

        scn_id = f"auto-{len(scenarios):05d}"
        scenarios.append(
            {
                "id": scn_id,
                "engine": "ecb",
                "test_label": f"{spec['r_and_d_regime']}-{spec['threshold_type']}-{spec['shock_timing']}",
                "hypothesis": (
                    f"Does pattern {spec['r_and_d_regime']} under "
                    f"{spec['threshold_type']} diffusion withstand "
                    f"{spec['shock_timing']} shocks?"
                ),
                "num_periods": defaults["num_periods"],
                "firms_init": [{"id": scn_id, "K_AI": 10.0, "U_f": 1.0,
                                "U_nf": 0.5, "H_nf": 0.3}],
                "ecb": {
                    "eta_decay": eta_decay,
                    "mobility_elasticity": mobility_eps,
                    "shared_pool": spec["shared_pool"] == "yes",
                    "tau_spillover": spill_int,
                },
                "params": {
                    "threshold": lut["threshold_map"][spec["threshold_type"]],
                    "shock": {
                        "drop_pct": shock_drop_pct,
                        "start_year": lut["shock_timing"][spec["shock_timing"]],
                        "duration": 5,
                    },
                    **invest_block,
                },
            }
        )
    return scenarios

# ── internal helpers (thresholds, capital paths, labour splits, …) ────
def _zero(_: int) -> float:                     # reusable λ 0
    return 0.0

def _smooth_threshold(t_star: int, width: float, height: float) -> Callable[[int], float]:
    return lambda t: height / (1.0 + math.exp(-(t - t_star) / width))

def _cliff_threshold(t_star: int, height: float) -> Callable[[int], float]:
    return lambda t: height if t >= t_star else 0.0

# capital invest rate s(t)
def _boom_bust_s(amplitude: float, bust_after: int, base: float) -> Callable[[int], float]:
    def s(t: int) -> float:
        return base + (amplitude if t < bust_after else -amplitude)
    return s

def _steady_s(invest_rate: float) -> Callable[[int], float]:
    return lambda _t: invest_rate

def _capital_from_s(s_func: Callable[[int], float], K0: float = 1.0) -> Callable[[int], float]:
    """Iterative cache so each K(t) computed once → O(T)."""
    cache: Dict[int, float] = {0: K0}
    def K(t: int) -> float:
        for τ in range(1, t + 1):
            if τ not in cache:
                cache[τ] = cache[τ - 1] * (1.0 + s_func(τ - 1))
        return cache[t]
    return K

# labour & R-D split
def _labor_funcs(
    rd_hi: float,
    rd_lo: float,
    style: str,
    switch_year: int,          # +++ NEW +++
) -> tuple[
    Callable[[int], float],
    Callable[[int], float],
    Callable[[int], float],
]:
    if style == "production_heavy":
        rd_before, rd_after = rd_lo, rd_hi      # low → high
    else:  # rd_heavy
        rd_before, rd_after = rd_hi, rd_lo      # high → low


    # +++ switch happens exactly at the scenario’s t_star +++
    share_for_rd = lambda t: rd_before if t < switch_year else rd_after
    total_labor = lambda _t: 1.0
    labor_final = lambda t: (1.0 - share_for_rd(t)) * total_labor(t)
    return labor_final, total_labor, share_for_rd

# x-value allocators 
def _x_symmetric(_t, _prev, _syn, _intan, _know, K):          # pylint: disable=unused-argument
    n = len(_prev) or 1
    return np.full(n, K / n)

def _x_knowledge_scaled(_t, _prev, _syn, _intan, know, K):
    n = max(1, int(round(know)))
    return np.full(n, K / n)

def _x_frozen(_t, prev, *_rest, K):
    return prev * (K / prev.sum())

_X_UPDATERS = {
    "symmetric": _x_symmetric,
    "knowledge_scaled": _x_knowledge_scaled,
    "frozen": _x_frozen,
}

def _subset_kwargs(cls, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return the sub-mapping whose keys match dataclass *cls* fields."""
    allowed = set(cls.__dataclass_fields__)           # type: ignore[attr-defined]
    return {k: v for k, v in cfg_dict.items() if k in allowed}

# ── main builder ───────────────────────────────────────────────────────
def _build_scenario(row: Dict[str, Any], defaults: Dict[str, Any]) -> ScenarioCfg:
    # merge parameter dicts 
    p = {**defaults, **row.get("params", {})}

    ecb_cfg_src     = {**defaults.get("ecb_params", {}), **row.get("ecb", {})}
    triage_cfg_src  = {**defaults.get("triage_params", {}), **row.get("triage", {})}

    ecb_params_obj    = ECBParams(**_subset_kwargs(ECBParams,    ecb_cfg_src))
    triage_params_obj = TriageParams(**_subset_kwargs(TriageParams, triage_cfg_src))


    # threshold-driven synergy & intangible 
    th = p["threshold"]
    if row["threshold"] == "smooth":
        syn_func = _smooth_threshold(th["t_star"], th["width"], th["height"])
    else:
        syn_func = _cliff_threshold(th["t_star"], th["height"])
    mirror_intang = p.get("intangible_mirror", True)
    intan_func = syn_func if mirror_intang else _zero

    # investment → capital & intangible spend
    if row["investment_path"] == "boom_bust":
        bb = p["boom_bust"]
        base_rate = p["invest_rate_base"]            
        s_func = _boom_bust_s(bb["amplitude"], bb["bust_after"], base=base_rate)

        trickle = p["intangible_trickle"]          
        intangible_invest = lambda t, bb=bb, tr=trickle: tr if t < bb["bust_after"] else 0.0
    else:
        st = p["steady"]
        s_func = _steady_s(st["invest_rate"])

        trickle = p["intangible_trickle"]            
        intangible_invest = lambda _t, tr=trickle: tr

    capital_func = _capital_from_s(s_func)

    #  labour split + shock timing
    lb = p["labor"]
    t_star = p["threshold"]["t_star"]             
    labor_func, total_labor_func, share_for_rd = _labor_funcs(
        lb["rd_share_high"],
        lb["rd_share_low"],
        row["labor_split"],
        switch_year=t_star,                        
    )

    if row["shock_timing"] != "none":
        sh = p["shock"]
        start, end, drop = sh["start_year"], sh["start_year"] + sh["duration"], 1.0 - sh["drop_pct"]
        orig_total = total_labor_func
        def shocked_total(t: int) -> float:
            base = orig_total(t)
            return base * drop if start <= t < end else base
        total_labor_func = shocked_total
        labor_func = lambda t, base=labor_func: (1.0 - share_for_rd(t)) * shocked_total(t)

    strength = p.get("skill_interaction_strength", 0.0)
    if strength > 0.0:
        skill_interaction_on = True
        def _skill_interaction(t: int,
                               syn_t: float | None = None,
                               intan_t: float | None = None,
                               k: float = strength) -> float:
            # compute_y_romer will pass syn_t / intan_t; fall back if None
            syn_val  = syn_t   if syn_t  is not None else syn_func(t)
            int_val  = intan_t if intan_t is not None else intan_func(t)
            return k * syn_val * int_val
    else:
        skill_interaction_on = False
        _skill_interaction   = lambda _t, _s, _i: 0.0

    #  assemble & return dataclass 
    return ScenarioCfg(
        id=row["id"],
        num_periods=p["num_periods"],
        alpha=p["alpha"],
        dt_integration=p["dt_integration"],
        growth_explosion_threshold=p["growth_explosion_threshold"],
        labor_func=labor_func,
        total_labor_func=total_labor_func,
        share_for_rd_func=share_for_rd,
        capital_func=capital_func,
        synergy_func=syn_func,
        intangible_func=intan_func,
        x_values_updater=_X_UPDATERS[row["x_updater"]],
        intangible_invest_func=intangible_invest,
        intangible_kappa=p.get("intangible_kappa", 1.0),
        intangible_Ubar=p.get("intangible_Ubar", 5.0),
        intangible_epsilon=p.get("intangible_epsilon", 0.5),
        skill_interaction_func=_skill_interaction,
        skill_interaction_on=skill_interaction_on,
        ecb_params=ecb_params_obj,
        triage_params=triage_params_obj,
        test_label=row.get("test_label", ""),
        hypothesis=row.get("hypothesis", ""),
        tags=tuple(row.get("tags", ())), 
    )


"""
1. What the file does, step-by-step

load_scenarios() opens scenarios.yaml, merges each scenario’s params: block with the global defaults: block, and passes the combined dictionary to a private builder.

_build_scenario() translates the raw parameters into callable building-blocks:

a) Threshold functions
Smooth → logistic curve using _smooth_threshold;
Cliff → step function via _cliff_threshold.
If intangible_mirror: true the same curve drives intangibles; otherwise intangibles stay at zero unless the YAML supplies something else.

b) Capital path
Boom-bust creates a reinvest-rate s(t) that flips sign after bust_after;
Steady uses a constant rate.
_capital_from_s then turns that rate into a cached capital stock K(t).

c) Labor split
_labor_funcs sets up three closures that, given a period t, return
– final-goods labour LY(t)
– total labour L(t)
– R & D share share_for_rd(t).

A built-in switch year (t_star from the threshold block) flips the share from the “before” level to the “after” level, matching the production_heavy vs rd_heavy patterns.
Optional early- or late-shock logic multiplies total labour by a drop factor for a fixed interval.

d)Intermediate-goods allocator
A dictionary maps the YAML key x_updater to one of three strategies:

symmetric — always splits capital equally across existing varieties

knowledge_scaled — number of varieties scales with rounded knowledge stock

frozen — keeps yesterday’s variety count and just resizes the bundle

e)Skill-interaction channel
If skill_interaction_strength is positive, a closure computes
interaction = k · synergy(t) · intangible(t); otherwise it returns zero.
The boolean flag skill_interaction_on records whether this extra term is in play.

f) Assemble the dataclass
The final ScenarioCfg captures:

scalar knobs (num_periods, alpha, dt_integration, explosion threshold, logistic parameters)

every callable required by the solver (labor_func, capital_func, synergy_func, etc.)

a forward-compatibility hook (intangible_invest_func) that Phase G uses when it converts intangibles from a “curve” to a proper stock flow.

g) Return a list
load_scenarios() yields a list of fully built configurations—typically 288 when you combine the axis choices in the reference YAML.
sim_runner.py simply loops over that list and never worries about defaults, validation, or numeric values.

2. How to use it

from scenarios import load_scenarios
scenarios = load_scenarios("scenarios.yaml")
for cfg in scenarios:
    results = compute_y_romer(
        num_periods=cfg.num_periods,
        alpha=cfg.alpha,
        labor_func=cfg.labor_func,
        capital_func=cfg.capital_func,
        synergy_func=cfg.synergy_func,
        intangible_func=cfg.intangible_func,
        x_values_updater=cfg.x_values_updater,
        total_labor_func=cfg.total_labor_func,
        share_for_rd_func=cfg.share_for_rd_func,
        dt_integration=cfg.dt_integration,
        growth_explosion_threshold=cfg.growth_explosion_threshold,
        intangible_kappa=cfg.intangible_kappa,
        intangible_Ubar=cfg.intangible_Ubar,
        intangible_epsilon=cfg.intangible_epsilon,
    )

That is all the boiler-plate you need—the ScenarioCfg already owns every moving part.

3. Configuration knobs available through YAML

Threshold axis – threshold: smooth | cliff plus numeric shape (t_star, width, height).

Investment axis – investment_path: boom_bust | steady with amplitude, bust_after, or invest_rate.

Labour-allocation axis – labor_split: production_heavy | rd_heavy and the two share levels rd_share_high, rd_share_low.

Shock axis – shock_timing: early | late | none with start_year, duration, drop_pct.

Intermediate-goods allocator – x_updater: symmetric | knowledge_scaled | frozen.

Intangibles logistic uplift – intangible_kappa, intangible_Ubar, intangible_epsilon.

Skill interaction – skill_interaction_strength (set to > 0 to activate).

Everything else—time horizon, Euler step, explosion cap, capital base rate—lives in defaults: and may be overridden per scenario if you choose.

4. Extending the system
Want a new capital pathway, a different labour shock, or a brand-new variety allocator?
Add a helper function in this file, map a fresh YAML key to it in the small lookup dictionary, and you are done.
The solver remains untouched because all complexity is hidden behind these callables.

In short, scenarios.py is the decoupling layer: messy economic assumptions stay in YAML; the solver sees a clean, type-safe interface.
"""