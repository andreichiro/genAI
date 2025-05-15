# Project Overview & Replication Guide  

## 1 . Quick start — zero-click reproduction

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate       # optional but recommended
pip install -r requirements.txt                         # pandas, numpy, matplotlib, etc.

# 🔁 Run the entire pipeline (≈ 1–2 min on a laptop)
make reproduce              # or:  python replicate.py --jobs 4


## 2. genAI.py deep dive

compute_y_romer(…)
This single routine can reproduce all of the model variants described, from a textbook Romer to AI-augmented versions with stochastic noise, adaptive time-steps, logistic intangible spill-overs and dynamic capital deepening.

2.1 Core objective

For each discrete period t = 0 … num_periods – 1 it computes

Y_new(t) =  [L_Y(t)]^(1-α) · Σ_i  x_i(t)^α with   0 < α < 1

This is subject to the resource constraint:
Σ_i x_i(t)  =  K(t)

And under the influence of optional channels:
Knowledge stock A(t) – grows through a user-supplied or built-in knowledge_updater;
Synergy and intangible scalars – can scale the aggregator Cobb-Douglas style, or via a logistic accelerator;

synergy_cobb_douglas_on / intangible_cobb_douglas_on toggle the multiplicative γ- and ζ-powers.
synergy_cd_exponent and intangible_cd_exponent are those exponents (defaults 0.2 and 0.3).

Capital accumulation – capital_updater can turn today’s output into tomorrow’s capital;

Skill interaction – set skill_interaction_on=True and pass a skill_interaction_func(t, synergy_t, intangible_t) that returns any non-negative scalar. The main loop multiplies Y by this factor each period (if you leave the flag False the extra call is completely skipped.)

All of these are pluggable callables. If you omit one, the function auto-fills a safe default so the solver never sees a None.

2.2 Parameter map (what you can configure)

Time horizon & technology:

num_periods (≥1) – length of the run.

alpha (0<α<1) – elasticity of substitution → also controls monopolistic-competition mark-up.

growth_explosion_threshold (default 10¹²) terminates the loop and raises GrowthExplosionError if either Y or A crosses it. Useful for sweeps that hit unstable corners of parameter space.

Numerical control:

dt_integration – Euler step if you use the simple internal capital/knowledge ODEs.

adaptive_dt – set True to auto-halve dt_integration whenever any state variable jumps by more than rel_change_tol (default 20 %).

dt_floor – lower bound so the loop cannot shrink forever.

rel_change_tol sets what “too large” means for adaptive Δt (default 0.20 ⇒ 20 %).

Stochastic ribbons:

monte_carlo_sigma – log-normal σ applied multiplicatively to Y_new(t) after the deterministic pass.
– growth_explosion (True/False) tells you if the run tripped the early-abort guard.
– final_x_values echoes the last intermediate-goods vector so you can inspect variety counts or pass it as an initial condition to another run.

Setting it to zero keeps the model deterministic (the default).

rng – an np.random.Generator or an integer seed for reproducible randomness.

Labour allocation:

labor_func(t) – returns final-goods labour L_Y(t) if you do not use the split helpers.

total_labor_func(t) + share_for_rd_func(t) – if provided, the engine itself splits labour into L_Y(t) and L_A(t) internally (needed for endogenous knowledge growth).

Capital:

capital_func(t) – an exogenous path for K(t).

s_invest_rate, delta_capital – if you prefer the classic law
K̇ = s·Y – δ·K without writing a custom updater.

capital_updater(t, K_prev, synergy, intangible, Y) – full control over accumulation,
e.g. to inject an intangible premium into investment efficiency.

delta_capital pairs with s_invest_rate inside the default

capital ODE → K̇ = s Y – δ K.
Leave both at 0 to switch the ODE off.

Knowledge:

knowledge_updater(t, A_prev, synergy, intangible, x_vals) – your own law of motion.

If you leave it None, the function auto-selects one of three built-ins:

A partial-AI growth law if you specified phi_ai_rd>0 or fraction_ai_rd_func.

The classic Romer law Ȧ = δ·A·L_A if delta_knowledge>0.

A no-op updater that keeps A constant.

delta_knowledge is the δ in the classic Romer law
Ȧ = δ A L_A and is only used when no custom knowledge_updater is provided.

Intermediate-goods vector:

x_values_updater(t, x_prev, synergy, intangible, A, K) – must return an array that sums to K.
If omitted we fall back to the symmetrical rule x_i = K/A.

x_values_init – starting vector (default [1.0]).

Note x_values_init lets you start with a non-uniform
vector (e.g., calibrated from data) instead of the default [1.0].

Aggregator choice

aggregator_mode in { "classic", "zeira", "ces" } – selects one of the three helper
functions at the end of the file:

Modes:
classic; zeira; ces

Classic: 	Standard Dixit–Stiglitz
Zeira: 	    Partial automation (Zeira 1998). See fraction_automated_func
CES: 	    Constant-Elasticity-of-Substitution. See rho (σ = 1 / (1–ρ))

fraction_automated_func(t) feeds β(t) into the Zeira or CES mode.

rho controls the elasticity in the CES variant
(σ = 1 / (1 – ρ)).

tfp_func(t) can override synergy and supply an arbitrary productivity time-series.

Synergy & Intangibles:

synergy_func(t) / intangible_func(t) – any non-negative scalar series.

synergy_cobb_douglas_on, synergy_cd_exponent – raise Y by synergy^γ.

Same trio for intangibles.

intangible_logistic_on, intangible_kappa, intangible_Ubar, intangible_epsilon – apply a bounded boost
Y ← Y·(1 + ε·logistic(intangible)).

Skill interaction:

If skill_interaction_on is True, the custom skill_interaction_func(t, synergy, intangibles)
can inject an additional multiplicative term.

The routine returns a dictionary with the full trajectory (if store_results=True) and every final state, or just the finals if you only care about end-points.

2.3. Aggregator helpers
aggregator_classic – implements the standard Romer aggregator
Y = A · L_Y^(1–α) Σ x_i^α. tfp_val lets you bolt in synergy as pure TFP.

aggregator_zeira – mirrors Zeira’s “partial task automation”: a share β(t) of capital substitutes for labour. Setting beta=None collapses to a Cobb–Douglas in K and L.

aggregator_ces – flexible CES with exponent ρ ≠ 0.
When ρ→0 the expression converges to Cobb–Douglas; when ρ<0 capital and labour are complements.

All three are pure functions: no side-effects, easy to unit-test.

3.3. Knowledge-only solvers
compute_knowledge_romer – stand-alone Euler integrator for the old Romer law Ȧ = δ · A · L_A. Useful for sanity-checking the embedded updater.

compute_partial_knowledge_AI and compute_full_knowledge_AI – two extensions that match the paper’s partial-AI and full-AI specifications:

Ȧ = δ · A^θ · [ L_A + γ·K_AI,R ]^η       (partial)
Ȧ = δ · A^θ · [ γ·K_AI,R ]^η             (full)

Both accept callables for K_AI,R, γ(t), synergy, intangibles, etc., then march forward with Euler steps of size dt.

Helper one-liners compute_A_dot_* concentrate the derivative so you can unit-test it separately.

3.5. Labour split utility

labor_split(…) takes total labour and a share-to-R&D function and returns period-by-period
L_Y / L_A (and, if you pass a knowledge_updater, it updates A(t) along the way).

It is essentially a pre-processor so you don’t have to write boiler-plate splitting logic in every scenario.

3.6. Convenience wrappers for AI scenarios

use_partial_ai – injects a composite labour term L̃ = L_Y + φ(t)·K_AI(t) into the classic aggregator, so you can model AI capital augmenting human labour without editing the core solver.

use_full_ai – extreme case where human labour is negligible; the wrapper feeds φ·K_AI as the “labour” input and sets L_Y = 0.

Both functions simply pre-configure the arguments and call compute_y_romer; they add no new maths.

3.7. Organisation-Capital module
compute_org_intang_capital(…) implements the perpetual-inventory method of Eisfeldt & Papanikolaou (2013) with the option to:

feed in raw SG&A and CPI data series (you provide the callables),
apply a logistic saturation to mimic decreasing returns when the intangible stock approaches a threshold Ū(t),
inject extra intangible investment outside SG&A (e.g. your Phase-G trickle).

The default invocation with kappa_func = 0 and extra_invest_func = 0 reduces to the textbook formula: 

OC_{t+1} = (1 – δ)·OC_t  +  SGA_{t+1}/CPI_{t+1}.

3.8. How to use in practice

Write or load scenario callables – import load_scenarios from scenarios.py; that returns a fully wired ScenarioCfg where every field is a ready-made lambda that ultimately points at the functions in this file.

Call the engine

from genAI import compute_y_romer
from scenarios import load_scenarios

cfg = load_scenarios()[0]               # grab first scenario
out = compute_y_romer(
        num_periods           = cfg.num_periods,
        alpha                 = cfg.alpha,
        labor_func            = cfg.labor_func,
        capital_func          = cfg.capital_func,
        synergy_func          = cfg.synergy_func,
        intangible_func       = cfg.intangible_func,
        x_values_updater      = cfg.x_values_updater,
        knowledge_updater     = None,            # or your own
        total_labor_func      = cfg.total_labor_func,
        share_for_rd_func     = cfg.share_for_rd_func,
        adaptive_dt           = True,
        monte_carlo_sigma     = 0.10,
        aggregator_mode       = "classic",
        growth_explosion_threshold = 1e10,
)

Note that setting adaptive_dt=False re-enables fixed Δt even if rel_change_tol is provided.


Inspect results

df = pandas.DataFrame(out["outputs"])
df.plot(x="t", y="Y_new", logy=True)

