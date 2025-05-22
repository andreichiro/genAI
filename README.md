
### Math Formalizations

**Macro Growth Model (Baseline):** In baseline scenarios, the code can simulate a textbook Romer growth model of expanding intermediate varieties. Final output each period ttt is produced via a **Cobb–Douglas/CES aggregator** of labor and intermediate inputs. Specifically, the code uses a constant-elasticity-of-substitution (CES) production function:

$Y(t)  =  A [ α K(t)ρ  +  (1−α) LY(t)ρ]1/ρ ⁣,Y(t) \;=\; A \,\Big[\,\alpha \,K(t)^{\rho} \;+\; (1-\alpha)\,L_Y(t)^{\rho}\Big]^{1/\rho}\!,$

$Y(t)=A[αK(t)ρ+(1−α)LY(t)ρ]1/ρ,$

where $LY(t)L_Y(t)LY(t)$ is final-good labor, $K(t)K(t)K(t)$ is the total capital devoted to intermediate goods, $0<α<10<\alpha<10<α<1$ is the capital share, $AAA$ is total factor productivity, and $σ=1/(1−ρ)\sigma = 1/(1-\rho)σ=1/(1−ρ)$ is the elasticity of substitution. 

In the special case $ρ→0\rho \to 0ρ→0$, this reduces to Cobb–Douglas $Y=A KαLY 1−αY = A\,K^\alpha L_Y^{\,1-\alpha}Y=AKαLY1−α$. 

The model enforces a resource constraint $∑ixi(t)=K(t) \sum_i x_i(t) = K(t)∑ixi(t)=K(t)$ on intermediate inputs, and each variety’s output $xi(t)x_i(t)xi(t)$ enters the aggregator with exponent $α\alphaα$. This represents a standard monopolistic competition setup where $1/(1−α)1/(1-\alpha)1/(1−α)$ is the markup.

**Extensions:** The baseline aggregator can be augmented by *synergy* or *intangible* factors that scale productivity. The code allows logistic or power-law adjustments (e.g., a multiplicative factor $1+γ(t)1+\gamma(t)1+γ(t)$ on $AAA$) to capture external effects. However, these macro-focused features are largely legacy; in the current refactor the primary emphasis is the micro-level evaluator-centric model described next.

### Evaluator-Centric Model

The **Evaluator-Centric Bottleneck (ECB)** model is a detailed microeconomic simulation of innovation where the **capacity to evaluate ideas** (rather than just generate ideas) is the limiting factor. It tracks multiple firms over time, modeling idea generation, screening, and implementation.

### Idea Generation and Knowledge Quality

Each firm iii has an **AI-driven idea generation capital** $KAI,iK_{AI,i}KAI,i$. In period $ttt$, firm iii draws a number of new project ideas $Ni,tN_{i,t}Ni,t$  from a Poisson process:

$Ni,t  ∼  Poisson(λpoisson⋅KAI,i),N_{i,t} \;\sim\; \text{Poisson}\big(\lambda_{\text{poisson}} \cdot K_{AI,i}\big),Ni,t∼Poisson(λpoisson⋅KAI,i),$

where $λpoisson\lambda_{\text{poisson}}λpoisson$ is a constant idea arrival rate per unit of $KAIK_{AI}KAI$. 

Thus, greater AI/R&D capital yields more ideas on average.

Each new idea has an uncertain quality or payoff $vvv$. We assume a Bayesian setup: prior belief $v∼N(μprior, τ2)v \sim N(\mu_{\text{prior}},\,\tau^2)v∼N(μprior,τ2)$ and observation s=v+εs = v + \varepsilons=v+ε with $noise ε∼N(0, σ2)\varepsilon \sim N(0,\,\sigma^2)ε∼N(0,σ2)$. 

The **prior mean** $μprior\mu_{\text{prior}}μprior$ represents current knowledge; critically, it can be raised by **knowledge spillovers** from other firms.


If $\overline{U}_{\text{nf},-i}$ denotes the average *non-fungible* evaluator capital of all other firms (more on $U_{\text{nf}}$ below), the model increases firm $i$’s prior mean by an externality term $\Omega$:

$$
\mu_{\text{prior},i}
=
\mu_{\text{prior},i}^{(0)}
+
\Omega\,\overline{U}_{\text{nf},-i}.
$$


where $τspillover∈[0,1]\tau_{\text{spillover}} \in [0,1]τspillover∈[0,1]$ is a spillover intensity parameter. Each firm adds $Ω$ to its $μprior\mu_{\text{prior}}μprior$, meaning that a higher industry-wide stock of evaluator talent improves the baseline quality of new ideas for everyone (a rising tide lifts all boats).

The idea generation draws $N∼Poisson(λ⋅K
AI$).

### Bayesian Update and Triage

When ideas arrive, the firm performs an **initial evaluation (triage)** before committing full screening resources. For each idea’s noisy signal sss, the model computes the **posterior quality distribution** via Bayesian update:

- Posterior mean: $μpost = τ2τ2+σ2 s + σ2τ2+σ2 μprior,\displaystyle \mu_{\text{post}} \;=\; \frac{\tau^2}{\tau^2+\sigma^2}\;s \;+\; \frac{\sigma^2}{\tau^2+\sigma^2}\;\mu_{\text{prior}},μpost=τ2+σ2τ2s+τ2+σ2σ2μprior,$
- Posterior variance: $σpost2 = τ2 σ2τ2+σ2 .\displaystyle \sigma_{\text{post}}^2 \;=\; \frac{\tau^2\,\sigma^2}{\tau^2+\sigma^2}\,. σpost2=τ2+σ2τ2σ2.$

The code implements these formulas for each new idea and then assigns a **triage score**:

$T  =  μpost  +  λexplore⋅σpost2 ,T \;=\; \mu_{\text{post}} \;+\; \lambda_{\text{explore}} \cdot \sigma_{\text{post}}^2~,T=μpost+λexplore⋅σpost2 ,$

where $λexplore\lambda_{\text{explore}}λexplore$ is a configurable weight ($λ > 0$ favors more uncertain ideas by adding variance, $λ < 0$ is more conservative). This score $T$ reflects a trade-off between an idea’s expected value and uncertainty.

The firm then **filters ideas** using a threshold on $T$. The threshold can be an absolute cutoff or a percentile of the scores. By default, the repository uses a percentile rule (e.g., keep the top X% of ideas). Ideas with $T$ below the threshold are discarded immediately (triage reject), saving evaluation capacity for better opportunities. The fraction of ideas that pass triage is recorded as *triage efficiency* (for instance, a triage efficiency of 0.3 means 30% of incoming ideas survived triage).

Accepted ideas (with their updated $(\mu_{\text{post}},\;\sigma_{\text{post}}^{2})$) are enqueued in a FIFO queue for detailed screening. Each queue entry carries the arrival time and posterior stats of an idea.

### Screening Capacity and Congestion

Each firm has a limited **screening/evaluation capacity** per period, denoted $\Psi_{\text{eff}}$ (effective ideas processed per period). This capacity is the key endogenous outcome of the firm’s resources. It depends on three factors:

- **Evaluator Capital:** The firm’s stocks of *fungible* and *non-fungible* evaluation capital, denoted $UfU_fUf and UnfU_{nf}Unf$ respectively, and its evaluator skill level  $HnfH_{nf}Hnf$. These are explained in the next section. Intuitively, $UfU_fUf$ might represent evaluation infrastructure or contractable resources (scalable with money), while $UnfU_{nf}Unf$ represents in-house evaluator talent which is tied to specific skilled workers (not easily fungible). The *skill* stock $HnfH_{nf}Hnf$ reflects the expertise of those evaluators.
- **Returns to Scale:** There are diminishing returns in how these capitals translate to throughput. The model defines **total evaluation capital** for a firm as:
    
    $Utot  =  Uf  +  ξ1 Unf Hnf ζskill .U_{\text{tot}} \;=\; U_f \;+\; \xi_1 \, U_{nf} \,H_{nf}^{\,\zeta_{\text{skill}}}\,.Utot=Uf+ξ1UnfHnfζskill.$
    
    Here $\xi_1$ is a weighting parameter for non-fungible capital and $\zeta_{\text{skill}}$ is a skill exponent. This formulation means $U_{nf}$ is amplified by the skill level of the evaluators (with diminishing returns if $0<\zeta_{\text{skill}}<1$). A firm with more or better-skilled evaluators has higher effective capital. However, $U_f$ contributes linearly (it might be, e.g., software or tools that scale without skill).
    
- **Congestion Externality:** If many firms have high evaluation capacity, they may face an **industry-wide congestion** in evaluating ideas (e.g. overlapping searches, competition for the “easy” ideas). The model captures this by reducing each firm’s effective throughput when rivals are strong. 

Let $\overline{U}_{-i}$ be the mean $U_{\text{tot}}$ of the other firms.
Then the *effective* throughput $\Psi_{\text{eff}}$ is given by:
- $Ψeff=1+ηcongestionU−iΨraw(Utot)$,
    
    $Ψeff  =  Ψraw(Utot) 1  +  ηcongestion  U‾−i  ,\Psi_{\text{eff}} \;=\; \frac{\Psi_{\text{raw}}(U_{\text{tot}})}{\,1 \;+\; \eta_{\text{congestion}}\;\overline{U}_{-i}\,}\,,$
    
    where $\eta_{\text{congestion}} \ge 0$ is a congestion sensitivity. If $\eta_{\text{congestion}}=0$, firms are independent. If $\eta_{\text{congestion}}>0$, a higher average rival capacity $\overline{U}{-i}$ *will **divide** a firm’s raw throughput $\Psi{\text{raw}}$* and thus slow down everyone as the industry scales up.
    

The **raw screening capacity** $\Psi_{\text{raw}}(U_{\text{tot}})$ represents the firm’s own capacity absent external congestion. The code offers two functional forms:

- **Logistic (S-shaped) capacity:** By default, $\Psi_{\text{raw}}$ follows a logistic function that saturates at a maximum $\psi_{\max}$. 

Specifically:

$$
\Psi_{\text{raw}}(U_{\text{tot}})
=
\psi_0
+
\frac{\psi_{\max}-\psi_0}
     {1 + \exp\!\bigl[-\kappa\,(U_{\text{tot}}-U^\*)\bigr]}
$$
    
    $Ψraw(Utot)  =  ψ0  +  ψmax⁡−ψ0 1+exp⁡[−κ (Utot−U∗)]  .\Psi_{\text{raw}}(U_{\text{tot}}) \;=\; \psi_0 \;+\; \frac{\psi_{\max}-\psi_0}{\,1 + \exp[-\kappa\,(U_{\text{tot}} - U^*)]\,}\,.$
    
    Here $\psi_0$ is a baseline throughput (even with minimal capital), $U^*$ is the midpoint $U_{\text{tot}}$ at which capacity is half-saturated, and $\kappa$ is the steepness of the curve. This logistic form captures **diminishing returns**: as $U_{\text{tot}}$ grows large, $\Psi_{\text{raw}}$ approaches $\psi_{\max}$.
    
- **Inverted-U capacity:** Alternatively, $\Psi_{\text{raw}}$ can be set to rise and then *decline* if $U_{\text{tot}}$ exceeds an optimal level (bureaucratic bloat). The code provides an inverted-U curve: $Ψraw(Utot)=ψ0+(ψmax−ψ0)(U∗Utot)exp(1−U∗Utot),$
    
$$
\Psi_{\text{raw}}(U_{\text{tot}})
=
\psi_0
+
(\psi_{\max}-\psi_0)
\left(\frac{U_{\text{tot}}}{U^\*}\right)
\exp\!\left(
      1-\frac{U_{\text{tot}}}{U^\*}
\right)
$$

which peaks at $$U_{\text{tot}}=U^*$$ and then declines. This option (selected by `psi_shape: "inv_u"`) reflects *over-investment inefficiency*: beyond a point, extra evaluators may slow the process (too many cooks in the kitchen).
    

Given $$U_{\text{tot}}$$ and chosen $\Psi_{\text{raw}}$ form, the effective capacity is finalized by applying the congestion factor as above. In summary, $$\Psi_{\text{eff}}$$ grows with a firm’s own evaluation capital but faces diminishing returns and is *dampened by rivals’ capacity*.

Finally, there is a evaluation cost curve: $cE=(1+κE)ϕ$

### Screening Accuracy

Besides throughput, each firm has an evaluation **accuracy** $\theta$ representing the probability an evaluated idea is correctly identified (for success or failure). The model factors accuracy into two components:

- A **capital-based accuracy** $\theta_{\text{cap}} = 1 - \exp(-,\xi_{\text{success}} ,U_{\text{tot}})$, which increases as total evaluation capital rises. Here $\xi_{\text{success}}$ is a success-rate slope parameter. This term implies diminishing gains: with more capacity, the chance to catch good ideas approaches 100% but never exceeds it.
- A **skill-based accuracy** $\theta_{\text{skill}} = 1 - \exp(-,\chi_{\text{skill}} ,H_{nf})$, which increases with evaluator human capital $H_{nf}$. $\chi_{\text{skill}}$ is a slope for skill effect. This reflects that more skilled evaluators are better at distinguishing high-quality ideas.

These are combined multiplicatively:

$θtotal  =  θcap×θskill ,\theta_{\text{total}} \;=\; \theta_{\text{cap}} \times \theta_{\text{skill}}\,,$

$θtotal=θcap×θskill,$

and $\theta_{\text{total}}$ is capped at 1.0 (perfect accuracy). The multiplicative form means both sufficient resources *and* skilled people are needed for high screening accuracy. For example, if either $U_{\text{tot}}$ or $H_{nf}$ is very low, $\theta$ will be low regardless of the other.

### Idea Queue Dynamics and Latency

Ideas that pass triage enter the firm’s **FIFO queue** to await screening. Each period, a firm can evaluate up to $\Psi_{\text{eff}}$ ideas from its queue. The simulation allows $$\Psi_{\text{eff}}$$ to be fractional; it will process the floor $\lfloor \Psi_{\text{eff}}\rfloor$ ideas for sure, and one more idea with probability equal to the fractional part. This way, an effective capacity of 3.7 ideas means 3 ideas definitely and a 70% chance of a 4th idea being processed.

For each idea evaluated, the code records a **latency** (wait time in queue) and removes it from the queue. If an idea waits too long in the queue, its creative value may decay (e.g. opportunity cost of delay). The model includes a **creativity decay** factor: if an idea of base value $$v_0$$ waited $$w$$ periods, a portion $v_0(1 - e^{-,\eta_{\text{decay}} w})$ of its value is lost by the time of evaluation. Here $\eta_{\text{decay}}$ is a decay rate parameter. The total lost value from all ideas in a period is tallied as *creativity loss*.

Any ideas still in queue simply carry over to the next period (with their current age). The code can also compute summary stats like mean and 95th-percentile of latency each period to quantify bottlenecks.

### Production and Payoff

When an idea is screened, if it passes, it can be turned into a final product that generates output/revenue `$Y$`. The model treats each period’s evaluated ideas as contributing to the firm’s output via a production function that combines the firm’s **AI capital** and its **evaluation labor**. Specifically, after evaluating ideas, each firm’s **quantity of output** is determined by a **micro CES production function**:

$Q=[ αmicro KAIρ  +  (1−αmicro) Ψeffρ]1/ρ,Q = \Big[\,\alpha_{\text{micro}} \,K_{AI}^\rho \;+\;(1-\alpha_{\text{micro}})\,\Psi_{\text{eff}}^\rho \Big]^{1/\rho},Q=[αmicroKAIρ+(1−αmicro)Ψeffρ]1/ρ,$

with parameters $αmicro,ρ\alpha_{\text{micro}},\rhoαmicro,ρ$ (defaults: $\alpha_{\text{micro}}=0.35$, $\rho=-0.5$). Here $K_{AI}$ acts like a “capital” input and $\Psi_{\text{eff}}$ (the number of ideas effectively evaluated this period, akin to effective labor) is the other input. This functional form means diminishing returns and an elasticity of substitution $\sigma = 1/(1-\rho)$ between AI and evaluation. If $\rho$ is negative (the default), $K_{AI}$ and evaluation capacity are **complements** – both are needed in balance to produce output.

Given the output quantity $Q$, the model applies an **inverse demand curve** to determine the price $P$. The demand curve is a downward-sloping function (a hyperbolic form):

$P(Q)  =  Pmax⁡ 1+(Q/Q12)η  ⁣,P(Q) \;=\; \frac{P_{\max}}{\,1 + (Q/Q_{\frac{1}{2}})^{\eta}\,}\!,$

$P(Q)=1+(Q/Q21)ηPmax,$

where $P_{\max}$ is the choke price (price at zero quantity) and $Q_{1/2}$ is the quantity at which price is half of $P_{\max}$. The exponent $\eta>0$ controls curvature (if $\eta=1$, it’s a rectangular hyperbola). This ensures larger output lowers the unit price. Each firm’s **revenue** (new output value) is then

$Y=P(Q)⋅Q :contentReference[oaicite:70]index=70,Y = P(Q)\cdot Q~:contentReference[oaicite:70]{index=70},Y=P(Q)⋅Q :contentReference[oaicite:70]index=70,$

which is automatically less than linear in `$Q$` due to price erosion. In the code, $Y$ is called `"Y_new"` per period.

The model can optionally overlay **costs**. For example, an SG&A (selling, general & administrative) cost as a fraction of revenue can be added from an external dataset, but by default we focus on output.

At the end of each period, the model logs each firm’s performance (output $$Y$$, market share, etc.). Market share is simply $Y_i / \sum_j Y_j$ for the period.

### Capital and Skill Accumulation Dynamics

After production, firms update their stocks of evaluation capital and skill for the next period. Each firm’s **tangible AI capital $$K_{AI}$$** is currently held fixed in the simulation (no endogenous investment in $$K_{AI}$$ in this version). However, **evaluator capital** and **skill** evolve:

- **Depreciation:** Both forms of evaluation capital depreciate at rate $\delta$. The code uses default $\delta_{U_f}=0.05$ and $\delta_{U_{nf}}=0.05$ (5% decay per period). So, $U_f \leftarrow (1-\delta_{U_f}) U_f$ and $U_{nf} \leftarrow (1-\delta_{U_{nf}}) U_{nf}$ each period after use. Intuitively, without reinvestment the firm’s evaluation capacity will slowly erode (e.g. due to obsolescence or wear and tear on processes).
- **Learning-by-Doing:** Evaluator *skill* improves with use. After processing ideas, the human capital $H_{nf}$ is updated as: $Hnf(t+1)=(1−δH)Hnf(t)+μlearningΨeff(t)$,
    
    $Hnf(t+1)  =  (1−δH) Hnf(t)  +  μlearning  Ψeff(t) ,H_{nf}^{(t+1)} \;=\; (1 - \delta_H)\,H_{nf}^{(t)} \;+\; \mu_{\text{learning}}\;\Psi_{\text{eff}}^{(t)}\,,$
    
    where $\delta_H$ is the skill depreciation (default 2% per period) and $\mu_{\text{learning}}$ is the learning-by-doing rate (default 0.05). 
    
    Thus, every idea evaluated increases the skill stock. This captures cumulative experience gains: if a firm evaluates more ideas, its evaluators become more effective over time. The term is linear in $\Psi_{\text{eff}}$ here (for simplicity), but the code is structured to allow more complex concave learning rules if needed.
    
- **Education Pipeline (New Talent):** The model can exogenously inject new evaluator talent via a training pipeline. If configured (education_lag > 0), an economy-wide **trainee stock** $$S(t)$$ is tracked. Each period, a fraction of trainees *enroll* into training and after a fixed lag they *graduate* as new evaluators. Specifically, each period:
    - Graduates: $Gt=pipeline t−ℓ$
    - New enrollment: $Et=enroll_rateS t−1+enroll_const$
    - Trainee stock updates: $St=(1−retire_rate)(St−1−Gt+Et)$
    
    The new graduates $G$ are then split equally among firms, *each firm’s* $U_{nf}$ increases by $G/\text{(number of firms)}$. This models an influx of fresh evaluators entering the industry and being hired evenly by firms. (In reality firms could compete for graduates, but here they share equally for simplicity.) By adjusting `education_lag`, `enroll_rate`, etc., the user can simulate scenarios with constrained or expanded future talent supply.
    

### Competition: Mobility, Entry & Exit

**Evaluator Mobility:** Non-fungible evaluator capital $UnfU_{nf}Unf$ is tied to human talent, which can be **poached** by competitors if wage incentives arise. The model includes an optional *talent mobility* mechanism. If enabled (mobility elasticity $\epsilon > 0$), at the end of each period firms that offered above-average evaluator wages gain talent at the expense of lower-wage firms. The formula:

$ΔUnf,i  =  ϵ (wi−w‾),\Delta U_{nf,i} \;=\; \epsilon \,\big(w_i - \overline{w}\big),ΔUnf,i=ϵ(wi−w),$

where $$w_i$$ is firm $$i$$’s average evaluator wage and $\overline{w}$ is the industry-wide average wage. Positive $\Delta U_{nf,i}$ means firm $i$ gains evaluator capital (hiring from others), negative means it loses some. The total gain across firms is zero, conserving the total $$U_{nf}$$ in the industry. 

In practice, implementing this requires estimating $w_i$ each period. (While the current code structure anticipates a `wage` output in the KPI dict, the default model does not yet endogenously set wages – a possible extension using market-clearing or an optimization routine). 

If using this feature, one could define $w_i$ proportional to marginal product of evaluators or based on backlog (a higher backlog might force a firm to raise wages). For now, mobility is a configurable option for experimenting with strategic effects (e.g., the *“talent poaching moat”* scenario sets $\epsilon=0.4$ to examine its impact).

**Firm Entry and Exit:** The model allows the number of firms to change over time, driven by market forces:

- **Exit:** A firm that performs poorly for an extended period will **exit** the market. The rule implemented: if a firm’s return on assets (ROA) is below a threshold for a consecutive streak, it exits. Specifically, the code checks if ROA (defined as $Y / K_{AI}$ for the firm) was below -5% for the last 3 periods. If so, the firm is removed (interpreted as going bankrupt or leaving the industry).
    
    ROA could be negative if firms incur costs; in our simplified output-only model, a negative threshold effectively means extremely low output relative to capital. The exit criterion thus clears out persistently unproductive firms. All their future ideas are discarded upon exit.
    
- **Entry:** New firms can **enter** when the market is growing. After each period, the model computes the **sector-wide output growth** $Δln⁡Ytot\Delta \ln Y_{\text{tot}}ΔlnYtot$. A positive growth can trigger entry of new firms. The rule: the number of entrants $E_t$ is Poisson-distributed with mean $λ=c⋅max⁡(0,Δln⁡Ytot)\lambda = c \cdot \max(0, \Delta \ln Y_{\text{tot}})λ=c⋅max(0,ΔlnYtot)$.
    
    The coefficient $c$ (default 0.2) controls how responsive entry is to growth. Thus, a 10% sector growth with $c=0.2$ yields $\lambda=0.02$ (2% chance of one entrant, etc.). If entry occurs, the new firm(s) start with an initial capital endowment specified by the scenario (or defaults: e.g. one unit of $K_{AI}$ and zero $U_f$, $U_{nf}$). The code creates new firm states accordingly. This entry process represents entrepreneurs being attracted to a booming market.
    
    Taken together, these entry and exit dynamics mean the model supports *endogenous industry structure*: in downturns, weak firms fail; in booms, new firms crowd in. This keeps the simulation realistic for long horizons.