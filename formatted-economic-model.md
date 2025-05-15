# AI Economic Growth Model: Equations, Parameters and Scenarios

## 1) Equations Overview

### 1.1. Classic Romer (No AI)
**Final Output (Old Romer):**

Y_old(t) = [L_Y(t)]^(1−α) ∑_{i=1}^{A(t)} [x_i(t)]^α, 0 < α < 1.

**Knowledge Production:**

Ȧ_old(t) = δA(t)L_A(t).

**Labor Constraint:**

L = L_Y(t) + L_A(t).

No AI means no additional φK_AI term in final output, and no γK_AI,R in knowledge production.

### 1.2. Partial vs. Full AI (Adopters)
**Final Output**

Partial AI (Years 1–8):

Y_T(t) = [L_Y(t) + φK_AI(t)]^(1−α) ∑_{i=1}^{A(t)} [x_i(t)]^α.

Full AI (Years 9–10):

Y_F(t) = [φK_AI(t)]^(1−α) ∑_{i=1}^{A(t)} [x_i(t)]^α, L_Y(t) ≈ 0.

**AI-Augmented Knowledge**

Partial:

Ȧ_T(t) = δA(t)^θ [L_A(t) + γK_AI,R(t)]^η.

Full:

Ȧ_F(t) = δA(t)^θ [γK_AI,R(t)]^η, L_A(t) ≈ 0.

### 1.3. Intangible Capital: U_t
U_{t+1} = U_t + I_{U,t} × 1/(1+e^(−κ(U_t−Ū))),

where Ū ≈ 5 is the synergy threshold, κ ≈ 1 is the slope, and I_{U,t} depends on the firm's intangible investment rate (related to R&D intensities above).

### 1.4. Lockout as a Risk Index
LockoutRisk_j(t) = 1/(1+exp{−β(ΔU_{j,t}−Δ*)})

where ΔU_{j,t} = U_leader(t) - U_j(t). If your intangible gap vs. the leader is large, risk → 1; if you catch up, risk → 0.

## 2) Parameter Calibration
We use your Summary Table to set R&D % and intangible investment for each category:

**Top Labs (Elite private R&D labs):**
- R&D ~20% of revenue (very high, typical for "top synergy").
- High skill λ ≈ 0.9.
- Year 0 intangible U_0 ≈ 4.3.

**Intermediate ("Average") Labs:**
- R&D ~4% (matches "median US" from your table).
- λ ≈ 0.5.
- Year 0 intangible U_0 = 2.0.

**Bottom Labs:**
- R&D ~1% or near 0% (since ~80% US firms do negligible R&D).
- λ ≈ 0.2.
- Year 0 intangible U_0 = 1.0.

**IT Sector:**
- R&D 13–20% (per your data: $182B / $900B revenues in some sub-sectors, etc.).
- λ ≈ 0.7.
- Year 0 intangible U_0 = 3.0.

**Usual US Sector (No AI):**
- Classic Romer, no φK_AI.
- R&D ~4%.
- Year 0 intangible can be 1.5–2.0, but it does not feed into synergy or partial→full AI. They remain on the old track. We simply track them as a "control group."

Within each category, we map R&D % → intangible investment I_{U,t}. For example:

- Top Labs: invests intangible = 0.20 × (Revenue).
- Average: invests intangible = 0.04 × (Revenue).
- Bottom: invests intangible = 0.01 × (Revenue).
- IT: invests intangible = 0.15 × (Revenue), etc.

(We scale these so that intangible invests from Year 0 onward in the logistic formula.)

## 3) Year 0 Observations

### 3.1. Innovation Outcomes at Year 0 (Empirical)
**Top Labs:**
- +44% discovered materials relative to pre-AI ⇒ DM_0 = 1.44.
- +39% patents ⇒ PT_0 = 1.39.
- +17% new products ⇒ NP_0 = 1.17.
- AI automates ~57% of idea-generation.
- Output vs. old model ~ +44% as well, so Y_0 = 1.44.

**Intermediate:**
- Possibly +5–10% net gains ⇒ DM_0 = 1.05, PT_0 = 1.03, NP_0 = 1.02, Y_0 = 1.05.

**Bottom:**
- Negligible ⇒ DM_0 = 1.00, PT_0 = 1.00, NP_0 = 1.00, Y_0 = 1.00.

**IT Sector:**
- Some partial AI usage ⇒ maybe DM_0 = 1.20, PT_0 = 1.15, NP_0 = 1.10, Y_0 = 1.20.

**Usual US (No AI):**
- Remains at the old baseline ⇒ DM_0 = 1.00, PT_0 = 1.00, NP_0 = 1.00, Y_0 = 1.00.

### 3.2. Intangible & Synergy at Year 0
- Top Labs: U_0 = 4.3, synergy φ_0 = 0.50, γ_0 = 0.50.
- Intermediate: U_0 = 2.0, synergy φ_0 = 0.20, γ_0 = 0.20.
- Bottom: U_0 = 1.0, synergy φ_0 = 0.05, γ_0 = 0.05.
- IT Sector: U_0 = 3.0, φ_0 = 0.35, γ_0 = 0.35.
- Usual US (No AI): synergy does not apply.

## 4) Iteration from Year 0 to Year 10

### 4.1. Partial AI (Years 1–8)
Each year, labs update:

U_{t+1} = U_t + I_{U,t}/(1+e^(−κ(U_t−5))).

φ_{t+1}, γ_{t+1} grow from low to near 1.0 as U crosses 5.

AI automation fraction grows from 57% at t = 0 to ~100% by Year 10.

Innovation (discovered materials, etc.) updates via a synergy-based factor.

Lockout Risk = logistic function of intangible gap vs. the best performer.

### 4.2. Full AI (Years 9–10)
Y_F = [φK_AI]^(1−α) ∑x_i^α, Ȧ_F = δA^θ [γK_AI,R]^η, L_Y, L_A ≈ 0.

### 4.3. No-AI Group
Stays in classic Romer: Ȧ_old = δAL_A.

No synergy jump. No intangible threshold.
We track their final output the same way we do in the standard Romer framework (some modest growth from knowledge accumulation, but no "AI multiplier").

## 5) Consolidated Multi-Category Table
Below is a condensed table. We show only the key columns to keep it readable. Each row is a year. For AI adopters (Top, Intermediate, Bottom, IT), we show synergy (φ,γ), intangible U(t), discovered materials (DM_t), etc. For the No-AI group, we show the old-model output as a reference.

Note: The numeric jumps are plausible examples once we tie intangible investment to the R&D intensities in your summary. In actual production usage, you would run a program (e.g., Python, R) that systematically updates these each year.

### Table Legend

- **DM** = Discovered Materials index (vs. pre-AI baseline).
- **PT** = Patent index.
- **NP** = New Product Lines index.
- **Y** = Final Output (vs. old model baseline).
- **Lockout** = A "risk index" or probability in [0,1]. For the No-AI group, we don't define lockout risk because they have chosen not to adopt AI.

For each AI adopter, we track partial AI (Years 1–8) → full AI (Years 9–10).

Below, an abbreviated 5-year snapshot (Years 0,3,5,8,10). You can easily expand it to all 11 rows (0–10).

### 5.1. Master Table (5 Snapshots)

| Category                   | Year | U(t) | (phi,gamma)| AI % | DM   | PT   | NP   | Y    | Lockout Risk |
|----------------------------|------|------|------------|------|------|------|------|------|--------------|
| **Top Labs (Partial->Full AI)** |
|                            | 0    | 4.3  | (0.50,0.50)| 57%  | 1.44 | 1.39 | 1.17 | 1.44 | 0.00         |
|                            | 3    | 6.0  | (0.75,0.75)| 75%  | 2.05 | 1.85 | 1.42 | 1.90 | 0.00         |
|                            | 5    | 8.0  | (0.88,0.88)| 85%  | 2.60 | 2.30 | 1.70 | 2.30 | 0.00         |
|                            | 8    | 10.0 | (0.98,0.98)| 97%  | 3.50 | 3.20 | 2.35 | 2.90 | 0.00         |
|                            | 10   | 11.0 | (1.00,1.00)| 100% | 4.30 | 4.00 | 3.00 | 3.50 | 0.00         |
| **Intermediate Labs (Partial->Full AI)** |
|                            | 0    | 2.0  | (0.20,0.20)| 57%  | 1.05 | 1.03 | 1.02 | 1.05 | 0.60 (vs top)|
|                            | 3    | 3.5  | (0.35,0.35)| 70%  | 1.15 | 1.10 | 1.06 | 1.12 | 0.64         |
|                            | 5    | 5.0  | (0.60,0.60)| 80%  | 1.40 | 1.25 | 1.15 | 1.35 | 0.65↓        |
|                            | 8    | 7.8  | (0.85,0.85)| 95%  | 2.00 | 1.70 | 1.45 | 1.80 | 0.50         |
|                            | 10   | 9.2  | (0.95,0.95)| 99%  | 2.40 | 2.00 | 1.75 | 2.20 | 0.30         |
| **Bottom Labs (Partial->Full AI)** |
|                            | 0    | 1.0  | (0.05,0.05)| 57%  | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 (vs top)|
|                            | 3    | 1.3  | (0.08,0.08)| 65%  | 1.02 | 1.01 | 1.00 | 1.03 | 0.85         |
|                            | 5    | 1.5  | (0.10,0.10)| 75%  | 1.05 | 1.02 | 1.01 | 1.06 | 0.86         |
|                            | 8    | 1.8  | (0.13,0.13)| 90%  | 1.08 | 1.05 | 1.03 | 1.10 | 0.80         |
|                            | 10   | 2.0  | (0.15,0.15)| 100% | 1.12 | 1.07 | 1.05 | 1.15 | 0.75         |
| **IT Sector (Partial->Full AI)** |
|                            | 0    | 3.0  | (0.35,0.35)| 57%  | 1.20 | 1.15 | 1.10 | 1.20 | 0.30 (vs top)|
|                            | 3    | 5.0  | (0.65,0.65)| 75%  | 1.50 | 1.35 | 1.20 | 1.50 | 0.20         |
|                            | 5    | 7.2  | (0.85,0.85)| 85%  | 1.90 | 1.60 | 1.40 | 1.80 | 0.15         |
|                            | 8    | 9.5  | (0.97,0.97)| 97%  | 2.50 | 2.10 | 1.80 | 2.20 | 0.10         |
|                            | 10   | 10.5 | (1.00,1.00)| 100% | 3.00 | 2.50 | 2.10 | 2.80 | 0.05         |
| **Usual US Sector (No AI, classic Romer)** |
|                            | 0    | N/A  | N/A        | N/A  | 1.00 | 1.00 | 1.00 | 1.00 | N/A          |
|                            | 3    | N/A  | N/A        | N/A  | 1.07 | 1.05 | 1.04 | 1.08 | N/A          |
|                            | 5    | N/A  | N/A        | N/A  | 1.15 | 1.10 | 1.08 | 1.16 | N/A          |
|                            | 8    | N/A  | N/A        | N/A  | 1.30 | 1.20 | 1.15 | 1.33 | N/A          |
|                            | 10   | N/A  | N/A        | N/A  | 1.45 | 1.30 | 1.25 | 1.50 | N/A          |

**Notes on the Table:**

- Lockout Risk for each AI-adopter category is measured vs. Top Labs intangible. If the intangible gap is large, risk is closer to 1. If the gap narrows, risk drops.
- Bottom Labs never meaningfully raise U; synergy φ ≈ 0.15 by Year 10. So their final output is ~1.15, meaning +15% from pre-AI baseline. Their lockout risk is still high (~0.75).
- IT Sector invests intangible heavily, crossing Ū = 5 by Year 3. They eventually achieve synergy near 1.00 by Year 10, output ~2.80 vs. the old baseline.
- Usual US (No AI) grows via standard Romer knowledge accumulation; final output up to ~1.50 by Year 10, which is less than top synergy labs or advanced IT can achieve with partial→full AI.

## SCENARIO (1): Vary κ in the S-Curve
We compare Low κ = 0.5 vs. High κ = 2. The higher κ is, the more abrupt the intangible jump (and thus synergy jump) after crossing Ū = 5.

**Setup for the Example:**
- We use a "Top Lab" style investment: R&D ~20% of revenue, skill λ = 0.9.
- Initial intangible U_0 = 4.0.
- Each year, intangible invests an amount scaled to 20% of revenue. We do a short iteration for Years 0–5 to illustrate how intangible & synergy differ.
- We produce two side-by-side tables, each with 6 rows (Years 0–5).
- Column ΔU shows how intangible increments each year, φ_t is synergy for final output, Y(t) is final output vs. a pre-AI baseline of 1.0.

### (1A) Low κ = 0.5 (Smooth Accumulation)

| Year | U(t) | Increment ΔU                                      | phi_t | Output Y_t |
|------|------|--------------------------------------------------|-------|------------|
| 0    | 4.0  | --                                               | 0.45  | 1.40       |
| 1    | 4.4  | 0.4 = I_U,1 * [1 / (1 + e^(-0.5(4.0-5)))]        | 0.50  | 1.50       |
| 2    | 4.8  | 0.4                                              | 0.55  | 1.60       |
| 3    | 5.2  | 0.4                                              | 0.60  | 1.70       |
| 4    | 5.7  | 0.5 (once above 5, bigger increment)             | 0.70  | 1.85       |
| 5    | 6.4  | 0.7                                              | 0.75  | 2.00       |

Because κ = 0.5 is low, the logistic slope near U = 5 is not too steep. The intangible increments increase moderately after crossing 5. By Year 5, intangible = 6.4, synergy φ = 0.75. Output ~2.00× baseline. Gains are meaningful but relatively "smooth."

### (1B) High κ = 2 (Abrupt Leap)

| Year | U(t) | Increment ΔU                                      | phi_t | Output Y_t |
|------|------|--------------------------------------------------|-------|------------|
| 0    | 4.0  | --                                               | 0.45  | 1.40       |
| 1    | 4.3  | 0.3 = I_U,1 * [1/(1 + e^(-2(4.0-5)))]            | 0.48  | 1.48       |
| 2    | 4.7  | 0.4                                              | 0.52  | 1.55       |
| 3    | 5.3  | 0.6 (once near 5, bigger jump)                   | 0.65  | 1.75       |
| 4    | 7.0  | 1.7 (huge leap after crossing threshold)         | 0.90  | 2.30       |
| 5    | 8.4  | 1.4                                              | 0.95  | 2.50       |

As soon as U crosses 5, the logistic factor 1/[1+e^(-2(U-5))] shoots up. So intangible jumps from 5.3 → 7.0 in just 1 year. Synergy leaps from 0.52 to 0.65, then to 0.90. Output soars from 1.55 to 2.30 in one year.

**Comparison:** High-κ yields a more "winner-takes-most" dynamic. Once you pass the threshold, intangible & synergy accelerate dramatically, creating big gaps between labs that cross early vs. those that remain stuck at U < 5.

## SCENARIO (2): Split AI for Production vs. R&D
We now partition K_AI(t) into (phi × K_AI,prod) for production vs. (gamma × K_AI,R) for R&D. We want to see how final output vs. knowledge growth changes if we shift that fraction. Below is a 3-case example for a single lab over 5 years (partial AI). We track final output and knowledge index (A) at Year 5.

**Base:** α = 0.5, δ = 0.05, θ = η = 1 for simplicity.

**Year 0:** A_0 = 1.0, intangible U_0 = 3.5. Each year, the lab invests intangible and grows synergy from 0.40 → 0.90 if crossing threshold. K_AI(t) is fixed growth from 100 → 200 by Year 5 (example). We just vary fraction to production vs. R&D.

### Table: Splitting AI Capital

| Case | %K_AI for Prod vs. R&D                                | Y_5  | A_5  |
|------|-------------------------------------------------------|------|------|
| (A)  | 80% Prod, 20% R&D                                     | 2.10 | 1.30 |
|      | phi*K_AI,prod=0.8K_AI, gamma*K_AI,R=0.2K_AI           |      |      |
| (B)  | 50% Prod, 50% R&D                                     | 1.90 | 1.60 |
|      | phi*K_AI,prod=0.5K_AI, gamma*K_AI,R=0.5K_AI           |      |      |
| (C)  | 20% Prod, 80% R&D                                     | 1.65 | 1.90 |
|      | phi*K_AI,prod=0.2K_AI, gamma*K_AI,R=0.8K_AI           |      |      |

**Interpretation:**
- (A) yields highest final output short-run (Year 5) but only moderate knowledge.
- (C) invests more AI in R&D → bigger knowledge (A=1.90) but smaller immediate Y=1.65.
- (B) is intermediate.

**Policy Relevance:** Over-weighting production AI can yield immediate gains but hamper long-run knowledge expansion and future innovation. Over-weighting R&D can pay off eventually, but short-run output is smaller.

## SCENARIO (3): Zero–One Labor Shock at a Surprise Date
We forcibly remove almost all labor at Year 5 (instead of planned Year 9–10). Two labs:

- **Lab A:** Intangible ~4.8 by Year 4, nearly crossing threshold Ū = 5.
- **Lab B:** Intangible ~2.5 by Year 4, far below threshold.

At Year 5 exactly, labor L_Y, L_A ≈ 0. We see final output at Year 6:

### Table: Surprise Full-AI at Year 5

| Lab | Year4 U(4) | Year4 phi(4) | Output(4) | Year5:Labor=0 | Year6 phi(6) | Output(6) |
|-----|------------|--------------|-----------|---------------|--------------|-----------|
| A   | 4.8        | 0.55         | 1.65      | Switch->FullAI| 0.90         | 2.40      |
| B   | 2.5        | 0.25         | 1.10      | Switch->FullAI| 0.10         | 0.60      |

**Explanation:**
- Lab A, intangible near 5 => synergy leaps to ~0.90 => output 2.40.
- Lab B, intangible only 2.5 => synergy ~0.10 => output collapses to 0.60.

This scenario shows how timing can produce abrupt winners vs. losers. Labs near intangible threshold can skyrocket once labor is removed and AI capital takes over; labs below threshold cannot exploit AI effectively.

## SCENARIO (4): Sectoral Full AI Only
We have two sectors:
- **Software Sector:** switches to full AI at Year 5.
- **Manufacturing Sector:** stays partial or old Romer.

Assume both start Year 0 with intangible ~2.5, but software invests intangible heavily (~15% R&D), manufacturing invests ~3%. We check intangible, synergy, output at Year 5 & Year 10.

### Table: Sectoral Differences

| Sector                  | R&D% | Year5 U(5) | Year5 phi(5) | Year5Out | Year10 U(10) | Year10 phi(10) | Year10Out |
|-------------------------|------|------------|--------------|----------|--------------|----------------|-----------|
| Software(Full AI@5)     | 15%  | 5.1        | 0.65         | 1.60     | 8.5          | 1.00           | 2.80      |
| Manufacturing(Partial)  | 3%   | 3.0        | 0.20         | 1.10     | 4.0          | 0.35           | 1.30      |

At Year 5, software intangible hits 5.1 => synergy ~0.65 => they jump to full AI => output=1.60. By Year 10, intangible=8.5 => synergy=1.0 => output=2.80.

Manufacturing invests intangible lightly => synergy remains 0.20–0.35 => output ~1.30. Skilled labor/capital also flows to software, deepening the gap.

## SCENARIO (5): Stop–and–Go Intangible Investment
We simulate a "boom–bust" intangible pattern vs. a "steady" pattern from Year 0–8. Each invests the same total intangible over 8 years, but timing differs:

- **Boom–Bust:** invests heavily in Years 1–2, then zero in Years 3–5, partial in Years 6–8.
- **Steady:** invests a moderate, consistent amount each year.

We show intangible each year, synergy, output at Year 8. Both labs start with U_0 = 2.0, Ū = 5, κ = 1.

### Table: Stop-and-Go vs. Steady

| Year | Boom–Bust(U_t) | Steady(U_t)  | phi_BB vs phi_Steady | Output(8)     |
|------|----------------|--------------|----------------------|---------------|
| 0    | 2.0 (start)    | 2.0 (start)  | 0.20 vs 0.20        | Both ~1.05    |
| 1    | 3.0 (heavy)    | 2.4 (mod)    | 0.28 vs 0.23        |               |
| 2    | 4.2 (heavy)    | 2.8 (mod)    | 0.40 vs 0.28        |               |
| 3    | 4.2 (0 invest) | 3.2 (mod)    | 0.40 vs 0.32        |               |
| 4    | 4.2 (0 invest) | 3.7 (mod)    | 0.40 vs 0.38        |               |
| 5    | 4.2 (0 invest) | 4.3 (mod)    | 0.40 vs 0.45        |               |
| 6    | 4.6 (modest)   | 4.9 (mod)    | 0.45 vs 0.55        |               |
| 7    | 5.2 (modest)   | 5.4 (mod)    | 0.60 vs 0.65        |               |
| 8    | 5.8 (modest)   | 6.0 (steady) | 0.70 vs 0.75        | 1.80 vs 1.90  |

Boom–Bust crosses the threshold at Year 7 (U = 5.2) => synergy leaps from 0.45 to 0.60.

Steady crosses threshold slightly earlier (U ≈ 5.4 at Year 7) => synergy ~0.75 by Year 8 => final output 1.90 vs. 1.80.

The difference might not be huge if both eventually cross threshold, but the steady approach yields a faster synergy ramp and slightly higher Year 8 output. If the bust period is longer (e.g., zero intangible from Years 3–7), the boom–bust lab might never cross threshold, creating a big gap. The details depend on how large and how long the "bust" is.
