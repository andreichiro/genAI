# genAI.py

import numpy as np
from typing import Callable, Dict, Any, Optional, Union


class GrowthExplosionError(RuntimeError):
    """Raised when Y_new or knowledge blows past the safety threshold."""
    pass

def compute_y_romer(
    num_periods: int,
    alpha: float,
    adaptive_dt: bool = False,                 # auto-halve Δt if jump too large  
    rel_change_tol: float = 0.20,              # 20 % default trigger             
    dt_floor: Optional[float] = None,          # smallest allowed Δt              
    monte_carlo_sigma: float = 0.0,            # 0  = deterministic               
    rng: Union[int, np.random.Generator, None] = None,
    labor_func: Optional[Callable[[int], float]] = None,
    capital_func: Optional[Callable[[int], float]] = None,
    capital_updater: Optional[Callable[[int, float, float, float, float], float]] = None,
    synergy_func: Optional[Callable[[int], float]] = None,
    intangible_func: Optional[Callable[[int], float]] = None,
    knowledge_updater: Optional[Callable[[int, float, float, float, np.ndarray], float]] = None,
    x_values_updater: Optional[Callable[[int, np.ndarray, float, float, float, float], np.ndarray]] = None,
    x_values_init: np.ndarray = None,
    knowledge_init: float = 0.0,
    capital_init: float = 1.0,
    total_labor_func: Optional[Callable[[int], float]] = None,
    share_for_rd_func: Optional[Callable[[int], float]] = None,
    delta_knowledge: float = 0.0,
    dt_integration: float = 1.0,   # time-step for Euler if no custom knowledge_updater/capital_updater
    s_invest_rate: float = 0.0,    # 's' for capital invests if no user capital_updater
    delta_capital: float = 0.0,    # depreciation rate if no user capital_updater
    store_results: bool = True,

    aggregator_mode: str = "classic",
    fraction_automated_func: Optional[Callable[[int], float]] = None,
    rho: float = -1.0,
    tfp_func: Optional[Callable[[int], float]] = None,
    fraction_ai_rd_func: Optional[Callable[[int], float]] = None,
    phi_ai_rd: float = 0.0,
    growth_explosion_threshold: float = 1e12,

    synergy_cobb_douglas_on = True,
    synergy_cd_exponent = 0.2,
    intangible_cobb_douglas_on = True,
    intangible_cd_exponent = 0.3,

    intangible_logistic_on = True,
    intangible_kappa = 1.0,
    intangible_Ubar = 5.0,
    intangible_epsilon = 0.5,

    skill_interaction_func: Optional[Callable[[int, float, float], float]] = None,
    skill_interaction_on: bool = False

) -> Dict[str, Any]:
    """
    Runs a multi-period simulation of the 'full Romer' style model with the updated formula:

      Y_new(t) = [LY(t)]^(1 - alpha) * sum_{i=1..A(t)} [ x_i(t) ]^alpha,   (0< alpha <1)

      1) The resource constraint sum_{i=1..A(t)} x_i(t) = K(t).
         - This ensures intermediate goods usage is not 'free'; 
           the user must specify how x_i are updated to match K(t).
      2) Optionally, capital can accumulate over time via a user-supplied capital_updater,
         e.g.  K_{t+1} = K_t + dt*[ s*Y_new(t) - deltaK*K_t ].
      3) synergy(t), intangible(t), knowledge(t) placeholders remain, 
         so advanced expansions can shape x-values or capital invests, 
         but they do not forcibly appear in Y_new(t).
      4) Implicit markup & monopolistic competition are recognized 
         by retaining alpha in (0,1).

    Steps for each t in [0..num_periods-1]:
      1) synergy(t), intangible(t), knowledge(t) updated if user-coded or from last iteration
      2) capital(t) either from capital_func(t) (if static or user-coded) 
         or from capital_updater(...) if dynamic 
      3) If knowledge_updater is provided, knowledge_{t+1} = ...
      4) x_values_{t+1} = x_values_updater(...) ensuring sum(x_i)=K(t)
         or if none is provided, we do a symmetrical approach:
             sum(x_i) = K(t) => each x_i = K(t)/A(t).
      5) Y_new(t) = [LY(t)]^(1-alpha)* sum( x_i^alpha )
      6) Possibly store step logs
      7) Move to next iteration


   aggregator_mode in {"classic", "zeira", "ces"}:
       - "classic": the original aggregator
       - "zeira": partial tasks approach (fraction_automated_func)
       - "ces": a CES aggregator with parameter rho

     If fraction_ai_rd_func or phi_ai_rd>0 is given (but knowledge_updater is None),
     we implement partial-AI knowledge growth automatically.

     If Y_new or knowledge exceed growth_explosion_threshold at any iteration,
     'growth_explosion' is set to True in the final output.
      

    :param num_periods: int >=1
        Number of discrete periods to simulate.
    :param alpha: float in (0,1)
        The elasticity exponent from Dixit–Stiglitz aggregator, 
        capturing the markup + elasticity of substitution among intermediates.
    :param labor_func: optional(t)->float
        If you want final-labor L_Y(t). If None, we treat L_Y(t)=1.0 or some default inside the code.
    :param capital_func: optional(t)->float
        If you want a static or user-coded approach for K(t). 
        If capital_updater is also used, we’ll do a more advanced approach 
        that updates capital each iteration. If both None, we treat capital=1.0 by default.
    :param capital_updater: optional(t, capital_prev, synergy, intangible, Y_new)->float
        If you want dynamic capital accumulation each step. 
        E.g. capital_{t+1}= capital_t + dt*(s*Y_new - deltaK*capital_t).
        If None, we skip capital accumulation and rely on capital_func or a default of K=1.0.

    :param synergy_func: function(t)->float
        synergy(t) >=0. 
    :param intangible_func: function(t)->float
        intangible(t) >=0.
    :param knowledge_updater: optional(t, knowledge_prev, synergy_t, intangible_t, x_vals)->float
        If provided, user can define how knowledge evolves each step. Must be >=0.
    :param x_values_updater: optional(t, x_vals_prev, synergy, intangible, knowledge, capital)->np.ndarray
        Must produce x_i(t+1) with sum_i x_i = capital(t). If None, we do symmetrical approach: 
          each x_i(t+1) = capital(t)/A(t). 
        For user expansions, must ensure nonnegative x_i.
    :param x_values_init: array-like
        initial x-values if needed. If None, default is [1.0].
    :param knowledge_init: float >=0
        initial knowledge. If no knowledge_updater, we ignore it.
    :param capital_init: float >=0
        initial capital. If no capital_updater, we only use capital_func or default=1.0.
    :param store_results: bool
        If True, store each step in 'outputs'. If False, only final states are returned.

    :return: dictionary with:
      {
        "outputs": list of step logs if store_results=True,
        "final_x_values": final x_i(t),
        "final_knowledge": final knowledge or None,
        "final_synergy": synergy_{end},
        "final_intangible": intangible_{end},
        "final_capital": capital_{end} if used,
        "final_Y_new": last computed Y_new(t)
      }
    :raises ValueError: 
        if alpha not in (0,1), synergy<0, intangible<0, capital<0, knowledge<0, x_values negative, etc.
    """

    # (1) Possibly define a partial-AI knowledge_updater if user gave fraction_ai_rd_func or phi_ai_rd>0
    def partial_ai_knowledge_updater(
        t: int,
        knowledge_prev: float,
        synergy_t: float,
        intangible_t: float,
        x_vals: np.ndarray
    ) -> float:
        # Real formula for partial AI approach:
        # 1) compute R&D labor S_rd if total_labor_func, share_for_rd_func
        if total_labor_func and share_for_rd_func:
            L_total = total_labor_func(t)
            frac_rd = share_for_rd_func(t)
            if L_total < 0 or frac_rd<0 or frac_rd>1:
                return knowledge_prev
            S_rd = L_total * frac_rd
        else:
            S_rd = 0.0

        # 2) fraction of R&D tasks done by AI:
        if fraction_ai_rd_func:
            frac_ai = fraction_ai_rd_func(t)
            if frac_ai<0: frac_ai=0
            if frac_ai>1: frac_ai=1
        else:
            frac_ai=0.0

        denom = 1.0 - frac_ai
        if denom < 1e-9:
            denom = 1e-9

        # 3) partial AI derivative:
        #    A_dot = phi_ai_rd * A(t)* (S_rd / denom)
        #    plus delta_knowledge if >0
        A_dot = (
            phi_ai_rd * knowledge_prev * (S_rd / denom)
            + delta_knowledge * knowledge_prev * S_rd
        )

        # 4) Euler step
        A_next = knowledge_prev + dt_integration * A_dot
        return max(A_next, 0.0)

    if knowledge_updater is None:
        if (phi_ai_rd > 0 or fraction_ai_rd_func is not None):
            # partial AI knowledge approach
            knowledge_updater = partial_ai_knowledge_updater
        elif delta_knowledge > 0.0:

            def enforced_knowledge_updater(
                t: int,
                knowledge_prev: float,
                synergy_t: float,
                intangible_t: float,
                x_vals: np.ndarray
            ) -> float:
                # We'll get L(t) from total_labor_func, fraction from share_for_rd_func => L_A(t)
                # If these funcs are missing, we fallback to 0.0 for L_A
                L_total = total_labor_func(t) if total_labor_func else 1.0
                frac_rd  = share_for_rd_func(t) if share_for_rd_func else 0.0
                L_A      = L_total * frac_rd
                # Euler step: A_{t+1} = A(t) + dt_integration*( delta_knowledge*A(t)*L_A )
                return knowledge_prev + dt_integration*(delta_knowledge * knowledge_prev * L_A)
            knowledge_updater = enforced_knowledge_updater
        else:
            # default knowledge_updater
            def no_op_knowledge_updater(
                t:int, knowledge_prev: float, synergy_t: float, intangible_t: float, x_vals: np.ndarray
            ) -> float:
                return knowledge_prev
            if knowledge_updater is None:
                knowledge_updater = no_op_knowledge_updater
      
        # user-supplied knowledge_updater
    # ENFORCE default capital_updater if none is supplied but s_invest_rate>0 or delta_capital>0
    if capital_updater is None and (s_invest_rate>0.0 or delta_capital>0.0):
        def enforced_capital_updater(
            t: int,
            capital_prev: float,
            synergy_t: float,
            intangible_t: float,
            Y_new: float
        ) -> float:
            # K_{t+1} = K_t + dt_integration*( s_invest_rate*Y_new - delta_capital*K_t )
            return capital_prev + dt_integration*(s_invest_rate*Y_new - delta_capital*capital_prev)
        capital_updater = enforced_capital_updater

    # ----------------------------------------------------------------
    # 0) Validate Basic Inputs
    # ----------------------------------------------------------------
    if not isinstance(num_periods, int) or num_periods < 1:
        raise ValueError(f"num_periods must be a positive integer, got {num_periods}")
    if not (0< alpha <1):
        raise ValueError(f"alpha must be in (0,1). got {alpha}")

    if capital_init <= 0.0:
            raise ValueError(
                "capital_init must be strictly positive; "
                f"got {capital_init}. Put the number in YAML instead of leaving the default."
            )
    # default labor or capital if user does not supply
    def default_labor_func(_t: int) -> float:
        return 1.0  # treat L_Y=1 if none
    if labor_func is None:
        labor_func = default_labor_func

    def default_capital_func(_t: int) -> float:
        return 1.0  # treat K=1 if none
    if capital_func is None and capital_updater is None:
        capital_func = default_capital_func

    # synergy, intangible checks at t=0
    synergy_0 = synergy_func(0)
    intangible_0 = intangible_func(0)
    if synergy_0 < 0:
        raise ValueError(f"synergy_func(0) returned negative synergy: {synergy_0}")
    if intangible_0 < 0:
        raise ValueError(f"intangible_func(0) returned negative intangible: {intangible_0}")

    # knowledge init checks
    if knowledge_init<0:
        raise ValueError(f"knowledge_init must be >=0, got {knowledge_init}")

    if dt_floor is None:
        dt_floor = dt_integration / 16.0           #  ⩗ sensible default  
        if dt_floor <= 0:
            dt_floor = 1e-6

    # reproducible RNG for Monte-Carlo ribbons
    if isinstance(rng, int) or rng is None:
        rng = np.random.default_rng(seed=rng)
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be int | None | np.random.Generator")


    # prepare x_values init
    if x_values_init is None:
        x_values_init = np.array([1.0], dtype=float)
    else:
        x_values_init = np.array(x_values_init, dtype=float)
        if any(x<0 for x in x_values_init):
            raise ValueError(f"x_values_init cannot contain negative entries. got {x_values_init}")

    # if user did not supply x_values_updater, define symmetrical approach
    def symmetrical_x_values_updater(
        t: int, x_vals_prev: np.ndarray, synergy_t: float, intangible_t: float, 
        knowledge_t: float, capital_t: float
    ) -> np.ndarray:
        """
        Splits capital_t equally among A(t) varieties (which is len(x_vals_prev) if user 
        does not dynamically change variety dimension).
        If user wants to expand or contract A(t), they'd do so in a custom x_values_updater.
        """
        # we assume A(t)= len(x_vals_prev)
        A_t = len(x_vals_prev)
        if A_t<=0:
            # fallback to single variety if none
            return np.array([capital_t], dtype=float)
        x_new = np.full(A_t, fill_value=capital_t/A_t, dtype=float)
        return x_new

    if x_values_updater is None:
        # partial function that does symmetrical approach
        def default_x_values_updater(
            t: int,
            x_vals_prev: np.ndarray,
            synergy_t: float,
            intangible_t: float,
            knowledge_t: float,
            capital_t: float
        ) -> np.ndarray:
            return symmetrical_x_values_updater(t, x_vals_prev, synergy_t, intangible_t, knowledge_t, capital_t)

        x_values_updater = default_x_values_updater
    else:
        # user-supplied must handle sum(x_i)=K(t) themselves
        pass

    # if capital_updater is provided, we track capital in dynamic form
    # if not, we rely on capital_func(t).
    capital_current = float(capital_init)

    # ----------------------------------------------------------------
    # 1) Initialize States
    # ----------------------------------------------------------------
    x_values_current = x_values_init.copy()
    knowledge_current = knowledge_init
    synergy_current = synergy_0
    intangible_current = intangible_0

    # For logging each step
    results=[]
    final_Y_new = None

    growth_explosion = False

    # 2) Main Iteration
    for t in range(num_periods):
        step_ok = False
        while not step_ok:
            # synergy, intangible each step
            synergy_t = synergy_func(t)
            intangible_t = intangible_func(t)
            if synergy_t<0:
                raise ValueError(f"synergy<0 at step {t}, synergy={synergy_t}")
            if intangible_t<0:
                raise ValueError(f"intangible<0 at step {t}, intangible={intangible_t}")

            # knowledge update
            knowledge_next = knowledge_updater(t, knowledge_current, synergy_t, intangible_t, x_values_current)
            if knowledge_next<0:
                raise ValueError(f"knowledge_updater produced negative knowledge at t={t}, {knowledge_next}")

            if capital_updater is None and capital_func is not None:
                capital_current = capital_func(t)

        # ENFORCE default knowledge_updater if none is supplied but user gave delta_knowledge>0
        # we only do "y_new" after we know labor(t) and x-values for the iteration
        # but x-values depends on capital, so let's get labor(t)
        LY_t=0.0
        if labor_func is not None:
            if total_labor_func and share_for_rd_func:
                L_t = total_labor_func(t)
                rd_frac = share_for_rd_func(t)
                if L_t < 0:
                    raise ValueError(f"total_labor_func returned negative L at step {t}")
                if rd_frac<0 or rd_frac>1:
                    raise ValueError(f"share_for_rd_func returned fraction not in [0,1] at step {t}")
                LA_t = L_t*rd_frac
                LY_t = L_t - LA_t
            else:
                # fallback: user-labor-func is final-labor directly, or default=1
                LY_t = labor_func(t) if labor_func else 1.0
                LA_t = 0.0
        # next we get capital for this iteration:
        # if capital_updater is used, we haven't updated capital yet 
        # from last iteration's Y. So let's do a 2-phase approach:
        #   1) We do x_values with "capital_current" from last iteration 
        #   2) compute Y_new for this iteration 
        #   3) then do capital_{t+1}= capital_current + dt*( s Y_new - deltaK capital_current ), etc.
        # This is consistent with "capital_{t+1}" depends on Y_new(t).
        # So let's do it:

        # 2a) x_values(t+1) using capital_current
        x_values_next = x_values_updater(
            t,
            x_values_current,
            synergy_t,
            intangible_t,
            knowledge_next,
            capital_current # enforce sum(x_i)= capital_current
        )
        if any(x<0 for x in x_values_next):
            raise ValueError(f"x_values_updater returned negative x_i at step {t}: {x_values_next}")

        # 2b) compute Y_new(t):
        # 1) fraction_automated => beta_t
        beta_t = None
        if fraction_automated_func:  # only relevant for "zeira"/"ces"
            raw_beta = fraction_automated_func(t)
            beta_t = max(0.0, min(1.0, raw_beta))  # clamp to [0,1]

        # 2) TFP from tfp_func or synergy:
        if tfp_func:
            A_tf = tfp_func(t)
            if A_tf < 0:
                A_tf = 0.0
        else:
            A_tf = 1.0 + synergy_t
            if A_tf < 0:
                A_tf = 0.0


        # 3) aggregator_mode:
        if aggregator_mode == "classic":
            # use aggregator_classic
            Y_new_t = aggregator_classic(
                alpha=alpha,
                labor_current=LY_t,           # final-labor for aggregator
                x_values=x_values_next,       # the updated x-values
                tfp_val=A_tf
            )
        elif aggregator_mode == "zeira":
            # use aggregator_zeira
            Y_new_t = aggregator_zeira(
                alpha=alpha,
                beta=beta_t,                  # fraction automated
                capital_current=capital_current,
                labor_current=LY_t,
                tfp_val=A_tf,
                synergy=synergy_t,
                intangible=intangible_t
            )
        elif aggregator_mode == "ces":
            # use aggregator_ces
            Y_new_t = aggregator_ces(
                capital_current=capital_current,
                labor_current=LY_t,
                beta=beta_t,
                tfp_val=A_tf,
                rho=rho,
                synergy=synergy_t,
                intangible=intangible_t
            )
        else:
            raise ValueError(f"aggregator_mode={aggregator_mode} not recognized.")

        if (Y_new_t > growth_explosion_threshold) or (knowledge_next > growth_explosion_threshold):
                    raise GrowthExplosionError(
                        f"Growth explosion at t={t}: Y_new={Y_new_t:.3e}, "
                        f"knowledge={knowledge_next:.3e}"
                    )

        if synergy_cobb_douglas_on and synergy_t > 0:
            Y_new_t *= (synergy_t ** synergy_cd_exponent)

        if intangible_cobb_douglas_on and intangible_t > 0:
            Y_new_t *= (intangible_t ** intangible_cd_exponent)


        # If intangible_logistic_on is True, do:
        if intangible_logistic_on:
            # logistic_factor in [0..1]
            logistic_factor = 1.0 / (1.0 + np.exp(-intangible_kappa * (intangible_t - intangible_Ubar)))
            # multiply aggregator by (1 + intangible_epsilon*logistic_factor)
            Y_new_t *= (1.0 + intangible_epsilon * logistic_factor)

        # final aggregator output after intangible expansions.
        final_Y_new = Y_new_t


        max_rel = 0.0
        if knowledge_current > 0:
            max_rel = max(max_rel,
                            abs(knowledge_next - knowledge_current)/knowledge_current)
        if final_Y_new and final_Y_new > 0:
            max_rel = max(max_rel,
                            abs(Y_new_t - final_Y_new) / final_Y_new)

        if adaptive_dt and max_rel > rel_change_tol and dt_integration > dt_floor:
            dt_integration /= 2.0      # halve step, retry
            continue                   # re-do this period with smaller Δt
        else:
            step_ok = True             # accept results


        # Monte-Carlo ribbons  
        if monte_carlo_sigma > 0.0:
            noise_factor = rng.lognormal(mean=0.0, sigma=monte_carlo_sigma)
            Y_new_t *= noise_factor
            final_Y_new = Y_new_t      # keep logs consistent

        #if we have capital_updater, we do capital_{t+1}
        if capital_updater is not None:
            # capital_{t+1}= capital_updater(t, capital_current, synergy, intangible, Y_new)
            capital_next= capital_updater(t, capital_current, synergy_t, intangible_t, Y_new_t)
            if capital_next<0:
                raise ValueError(f"capital_updater returned negative capital at step {t}: {capital_next}")
        else:
            # else we keep capital the same or get from capital_func(t+1)
            # but let's do "capital_next" = capital_func(t+1) if capital_func is not None
            if capital_func is not None:
                capital_next= capital_func(t+1)
                if capital_next<0:
                    raise ValueError(f"capital_func returned negative capital at step {t+1}")
            else:
                capital_next= capital_current  # default approach

            # store step results
            if store_results:
                results.append({
                    "t": t,
                    "LY": LY_t,
                    "LA": LA_t,
                    "capital_current": capital_current,
                    "x_values": x_values_next.copy(),
                    "synergy": synergy_t,
                    "intangible": intangible_t,
                    "knowledge_before": knowledge_current,
                    "knowledge_after": knowledge_next,
                    "Y_new": Y_new_t,
                    "growth_explosion_so_far": growth_explosion
                })

            # commit updates
            synergy_current = synergy_t
            intangible_current = intangible_t
            knowledge_current = knowledge_next
            x_values_current = x_values_next
            capital_current = capital_next

        # end loop

    return {
        "outputs": results if store_results else None,
        "final_x_values": x_values_current,
        "final_knowledge": knowledge_current,
        "final_synergy": synergy_current,
        "final_intangible": intangible_current,
        "final_capital": capital_current,
        "final_Y_new": final_Y_new,
        "growth_explosion": growth_explosion
    }

def aggregator_classic(
    alpha: float,
    labor_current: float,
    x_values: np.ndarray,
    tfp_val: float = 1.0,
    **kwargs
) -> float:
    """
    The 'classic' aggregator from the original code:
      Y = [labor_current]^(1 - alpha) * sum( x_i^alpha ).
    The parameter tfp_val can be multiplied in if desired, 
    but in the original 'classic' aggregator we often just treat synergy
    outside this formula. We keep it flexible.
    """
    sum_x_alpha = np.power(x_values, alpha).sum()
    labor_term = labor_current ** (1.0 - alpha)
    # multiply by tfp_val 
    return tfp_val * labor_term * sum_x_alpha

def aggregator_zeira(
    alpha: float,
    beta: Optional[float],
    capital_current: float,
    labor_current: float,
    tfp_val: float = 1.0,
    synergy: float = 1.0,
    intangible: float = 1.0,
    **kwargs
) -> float:
    """
    Zeira aggregator:
      If beta is None => Y = tfp_val * K^alpha * L^(1-alpha)
      else => partial fraction:
        Y = tfp_val * [ (beta*K)^alpha * ((1 - beta)*L )^(1 - alpha ) ].
    """
    if beta is None:
        # pure => Y = tfp_val*(K^alpha)*(L^(1-alpha))
        return tfp_val * (capital_current**alpha) * (labor_current**(1.0 - alpha))
    else:
        # partial => Y= tfp_val* [ (beta*K)^alpha * ((1 - beta)*L )^(1-alpha) ]
        K_part = (beta * capital_current)**alpha
        L_part = ((1.0 - beta) * labor_current)**(1.0 - alpha)
        return tfp_val * (K_part * L_part)

def aggregator_ces(
    capital_current: float,
    labor_current: float,
    beta: float,
    tfp_val: float,
    rho: float,
    synergy: float = 0.0,
    intangible: float = 0.0,
    **kwargs
) -> float:
    """
    Implements a Constant-Elasticity-of-Substitution (CES) aggregator in its
    more standard form:

        Y = tfp_val * [ beta * (K^rho) + (1 - beta) * (L^rho) ]^(1 / rho).

    The elasticity of substitution (sigma) is related to rho by:
        sigma = 1 / (1 - rho).
    If 0 < rho < 1, the elasticity is > 1 (goods are more easily substitutable);
    if rho < 0, the elasticity is < 1 (goods are less substitutable).
    If rho → 0, the expression can approach a Cobb–Douglas or log form,
    depending on the limit.

    Parameters
    ----------
    capital_current : float
        The capital quantity K(t) to include in the CES aggregator.
    labor_current : float
        The labor quantity L(t) to include in the CES aggregator.
    beta : float
        The share parameter in [0, 1] controlling the weights on K vs. L.
    tfp_val : float
        A multiplicative factor (A) acting as total factor productivity.
    rho : float
        The CES exponent. Must be != 0 for this form. Often < 1 or negative.
    synergy : float
        (Optional) synergy placeholder; not used unless you incorporate
        synergy into tfp_val or the aggregator. Default = 0.0.
    intangible : float
        (Optional) intangible placeholder; also not used unless integrated
        into the aggregator logic. Default = 0.0.
    **kwargs : dict
        Additional parameters for future extensions.

    Returns
    -------
    float
        The CES-aggregated output Y.

    Raises
    ------
    ValueError
        If beta is out of [0, 1], if inside_sum <= 0, or if any other
        numeric inconsistency arises.
    """

    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0,1]. Got {beta}")

    # Potentially, synergy/intangible could adjust tfp_val here if desired, e.g.:
    # tfp_val = tfp_val * (1.0 + synergy + intangible)

    # Compute partial contributions
    K_term = beta * (capital_current ** rho)
    L_term = (1.0 - beta) * (labor_current ** rho)
    inside_sum = K_term + L_term

    if inside_sum <= 0:
        # You could also raise an error if inside_sum < 0.
        return 0.0

    return tfp_val * (inside_sum ** (1.0 / rho))

def compute_knowledge_romer(
    num_periods: int,
    delta: float,
    # time-step for Euler integration
    dt: float,
    # Functions to supply dynamic inputs each period
    la_func: Callable[[int], float],         # R&D labor L_A(t)
    synergy_func: Callable[[int], float],    # synergy(t), if relevant
    intangible_func: Callable[[int], float], # intangible(t), if relevant
    # Additional logic for partial synergy or intangible-driven knowledge
    # can be done externally or in la_func, etc.
    # Initial condition for knowledge
    A_init: float = 0.0,
    # Whether we store step-by-step results
    store_results: bool = True
) -> Dict[str, Any]:
    """
    Runs a multi-period simulation of "old Romer knowledge" growth, defined by
        dA/dt = delta * A(t) * L_A(t),
    using a simple Euler integration with step size dt.

    For each t in [0..num_periods-1]:
      1) Retrieve synergy(t), intangible(t), and R&D labor L_A(t).
      2) Compute A_dot_old(t) = delta*A_current*L_A(t).
      3) Integrate: A_next = A_current + dt*A_dot_old(t).
      4) Validate that A_next >= 0 (nonnegative knowledge).
      5) Optionally store step results.

    synergy(t) and intangible(t) do not directly appear in the old formula 
    dA/dt = delta*A*L_A, but they can be used inside la_func(...) or 
    any external logic to shape L_A(t).

    :param num_periods: int, >=1
        Number of discrete steps to simulate.
    :param delta: float >= 0
        R&D productivity factor in the old knowledge formula.
    :param dt: float > 0
        Timestep used in the Euler integration (A_{t+1} = A_t + dt*A_dot).
    :param la_func: function(t)->float
        Returns R&D labor L_A(t) >=0 at each step. 
        This can incorporate synergy(t), intangible(t), or other user-defined logic if desired.
    :param synergy_func: function(t)->float
        synergy(t) >=0 each step. 
        If the old formula does not use synergy directly, you can still 
        rely on synergy_func inside la_func or other logic.
    :param intangible_func: function(t)->float
        intangible(t) >=0 each step.
    :param A_init: float >= 0
        Initial knowledge stock A(0).
    :param store_results: bool
        If True, stores each step’s details (A, synergy, intangible, L_A, A_dot, etc.).
        If False, only returns the final state.

    :return: Dict with:
        {
          "outputs": list of step results if store_results=True,
          "final_knowledge": A at the final step,
          "final_synergy": synergy at last step,
          "final_intangible": intangible at last step
        }
    :raises ValueError:
        If negative or invalid inputs are encountered (e.g. delta < 0, la_func(t)<0,
        synergy_func(t)<0, intangible_func(t)<0, dt<=0, etc.).
    """

    # ---------------------------------------------------------
    # 0) Validate Basic Parameters
    # ---------------------------------------------------------
    if not isinstance(num_periods, int) or num_periods < 1:
        raise ValueError(f"num_periods must be a positive integer, got {num_periods}.")
    if not (isinstance(delta, (float,int)) and delta >= 0):
        raise ValueError(f"delta must be a nonnegative float. Got {delta}")
    if not (isinstance(dt, (float,int)) and dt > 0):
        raise ValueError(f"dt must be a positive float. Got {dt}")
    if A_init < 0:
        raise ValueError(f"A_init (initial knowledge) must be >=0, got {A_init}")

    # We'll do synergy_func(0) etc. for reference
    synergy_current = synergy_func(0)
    intangible_current = intangible_func(0)
    if synergy_current < 0:
        raise ValueError(f"synergy_func(0) returned negative synergy.")
    if intangible_current < 0:
        raise ValueError(f"intangible_func(0) returned negative intangible.")

    # ---------------------------------------------------------
    # 1) Initialize State
    # ---------------------------------------------------------
    A_current = A_init
    results = []

    # ---------------------------------------------------------
    # 2) Main Loop
    # ---------------------------------------------------------
    for t in range(num_periods):
        # 2a) synergy, intangible, R&D labor
        synergy_t = synergy_func(t)
        intangible_t = intangible_func(t)
        LA_t = la_func(t)

        # Validate them
        if synergy_t < 0:
            raise ValueError(f"synergy_func returned negative synergy at step {t}: {synergy_t}")
        if intangible_t < 0:
            raise ValueError(f"intangible_func returned negative intangible at step {t}: {intangible_t}")
        if LA_t < 0:
            raise ValueError(f"la_func returned negative L_A(t) at step {t}: {LA_t}")

        # 2b) Compute A_dot = delta*A(t)*L_A(t)
        A_dot = compute_A_dot_romer(A_current, LA_t, delta)

        # 2c) Integrate: A_next = A_current + dt*A_dot
        A_next = A_current + dt*A_dot
        if A_next < 0:
            raise ValueError(f"Integration produced negative A(t+1) at step {t}. A_next={A_next}")

        # 2d) Optionally store step results
        if store_results:
            results.append({
                "t": t,
                "A_current": A_current,
                "A_dot": A_dot,
                "LA": LA_t,
                "synergy": synergy_t,
                "intangible": intangible_t,
                "A_next": A_next
            })

        # 2e) Move to next iteration
        A_current = A_next

    # ---------------------------------------------------------
    # 3) Final Output
    # ---------------------------------------------------------
    return {
        "outputs": results if store_results else None,
        "final_knowledge": A_current,
        "final_synergy": synergy_t,
        "final_intangible": intangible_t
    }

def compute_A_dot_romer(A: float, LA: float, delta: float) -> float:
    """
    Computes the old Romer knowledge derivative:
        A_dot_romer(t) = delta * A(t) * L_A(t).

    :param A: float, knowledge stock A(t) >= 0
    :param LA: float, R&D labor L_A(t) >= 0
    :param delta: float, R&D productivity >= 0
    :return: float, dA/dt
    :raises ValueError:
      if A<0, LA<0, or delta<0
    """
    if A < 0:
        raise ValueError(f"A(t) must be >=0, got {A}")
    if LA < 0:
        raise ValueError(f"L_A(t) must be >=0, got {LA}")
    if delta < 0:
        raise ValueError(f"delta must be >=0, got {delta}")

    return delta * A * LA

def labor_split(
    num_periods: int,
    total_labor_func: Callable[[int], float],
    share_for_rd_func: Callable[[int], float],
    knowledge_updater: Optional[Callable[
        [int, float, float, float, float, float], float
    ]] = None,
    synergy_func: Optional[Callable[[int], float]] = None,
    intangible_func: Optional[Callable[[int], float]] = None,
    knowledge_init: float = 0.0,
    store_results: bool = True
) -> Dict[str, Any]:
    """
    Splits total labor L(t) each period into final-labor L_Y(t) and R&D-labor L_A(t),
    and optionally updates knowledge if a user-provided `knowledge_updater` is given.

    Specifically:
        L_Y(t) = [1 - share_for_rd_func(t)] * L(t)
        L_A(t) = share_for_rd_func(t) * L(t)

    Parameters
    ----------
    num_periods : int
        Number of discrete time steps (t=0..num_periods-1).
    total_labor_func : Callable[[int], float]
        A function returning total labor L(t) for each period t.
    share_for_rd_func : Callable[[int], float]
        A function returning the fraction of labor allocated to R&D (0 <= fraction <= 1).
    knowledge_updater : Optional[Callable[[int, float, float, float, float, float], float]]
        A user-supplied function to update knowledge each step:
          knowledge_next = knowledge_updater(
              t,
              knowledge_prev,
              synergy_t,
              intangible_t,
              L_Y_t,
              L_A_t
          )
        If None, knowledge is not updated (remains at knowledge_init).
    synergy_func : Optional[Callable[[int], float]]
        A function returning synergy(t). If None, defaults to 0.0 each step.
    intangible_func : Optional[Callable[[int], float]]
        A function returning intangible(t). If None, defaults to 0.0 each step.
    knowledge_init : float
        Initial knowledge stock (nonnegative).
    store_results : bool
        If True, store step-by-step logs in "outputs". If False, only final state is returned.

    Returns
    -------
    Dict[str, Any]
        A dictionary with:
        - "outputs": a list of step records if store_results=True, else None
        - "final_knowledge": the knowledge stock after the last period
        - "final_synergy": synergy at the last period
        - "final_intangible": intangible at the last period
        - "final_LY": last computed L_Y
        - "final_LA": last computed L_A

    Raises
    ------
    ValueError
        If any function returns invalid values (e.g., negative labor),
        or if share_for_rd_func(t) is outside [0,1], etc.
    """

    # 1) Validate basic inputs
    if not isinstance(num_periods, int) or num_periods < 1:
        raise ValueError(f"num_periods must be a positive int, got {num_periods}.")
    if knowledge_init < 0:
        raise ValueError(f"knowledge_init must be nonnegative, got {knowledge_init}.")

    # 2) Provide default synergy/intangible if none
    if synergy_func is None:
        def synergy_func(_t: int) -> float:
            return 0.0
    if intangible_func is None:
        def intangible_func(_t: int) -> float:
            return 0.0

    # 3) Initialize state
    knowledge_current = knowledge_init
    LY_current, LA_current = 0.0, 0.0
    synergy_current = synergy_func(0)
    intangible_current = intangible_func(0)

    # For logging
    results = []

    # 4) Main loop
    for t in range(num_periods):
        # a) get synergy, intangible, total labor
        synergy_t = synergy_func(t)
        intangible_t = intangible_func(t)
        L_total = total_labor_func(t)
        frac_rd = share_for_rd_func(t)

        # Validate them
        if synergy_t < 0:
            raise ValueError(f"synergy_func returned negative synergy at t={t}: {synergy_t}")
        if intangible_t < 0:
            raise ValueError(f"intangible_func returned negative intangible at t={t}: {intangible_t}")
        if L_total < 0:
            raise ValueError(f"total_labor_func returned negative labor at t={t}: {L_total}")
        if not (0.0 <= frac_rd <= 1.0):
            raise ValueError(f"share_for_rd_func returned fraction not in [0,1] at t={t}: {frac_rd}")

        # b) split labor
        LY_t = (1.0 - frac_rd) * L_total
        LA_t = frac_rd * L_total

        # c) optionally update knowledge
        if knowledge_updater is not None:
            knowledge_next = knowledge_updater(
                t,
                knowledge_current,
                synergy_t,
                intangible_t,
                LY_t,
                LA_t
            )
            if knowledge_next < 0:
                raise ValueError(f"knowledge_updater produced negative knowledge at t={t}: {knowledge_next}")
        else:
            knowledge_next = knowledge_current

        # d) store results if desired
        if store_results:
            results.append({
                "t": t,
                "synergy": synergy_t,
                "intangible": intangible_t,
                "total_labor": L_total,
                "share_for_rd": frac_rd,
                "LY": LY_t,
                "LA": LA_t,
                "knowledge_before": knowledge_current,
                "knowledge_after": knowledge_next
            })

        # e) commit updates
        knowledge_current = knowledge_next
        synergy_current = synergy_t
        intangible_current = intangible_t
        LY_current = LY_t
        LA_current = LA_t

    # 5) Return final results
    return {
        "outputs": results if store_results else None,
        "final_knowledge": knowledge_current,
        "final_synergy": synergy_current,
        "final_intangible": intangible_current,
        "final_LY": LY_current,
        "final_LA": LA_current
    }

def use_partial_ai(
    num_periods: int,
    alpha: float,
    # user-provided functions that define partial AI
    labor_func: Callable[[int], float],    # L_Y(t)
    kai_func: Callable[[int], float],      # K_AI(t)
    phi_func: Callable[[int], float],      # phi(t)
    synergy_func: Callable[[int], float],
    intangible_func: Callable[[int], float],
    # Optional: knowledge_updater, x_values_updater, etc.
    knowledge_updater: Optional[Callable] = None,
    x_values_updater: Optional[Callable] = None,
    # initial states, etc.
    x_values_init: np.ndarray = None,
    knowledge_init: float = 0.0,
    capital_init: float = 0.0,
    store_results: bool = True
) -> Dict[str, Any]:
    """
    A convenience wrapper that sets up the partial-AI aggregator
    in compute_y_romer without rewriting everything.
    """
    # 1) We'll define a small "labor + AI" aggregator for Y_new.
    #    That aggregator is actually in x_values_updater logic or the final aggregator step.
    #    But in your generic 'compute_y_romer', the aggregator = [LY]^(1-alpha)* sum(x_i^alpha).
    #    So we must "trick" it to use (LY + phi*KAI) as the 'labor' term.

    # Step (A): define a custom labor_func that returns (LY + phi*KAI).
    def partial_ai_labor(t: int) -> float:
        LY_t = labor_func(t)
        phi_t = phi_func(t)
        kai_t = kai_func(t)
        return LY_t + phi_t * kai_t   # so 'labor' in aggregator is (LY + phi*KAI)

    # Step (B): define a capital_func that returns some static or default K=1,
    # or you can do something more advanced. If partial AI doesn't revolve around
    # a separate capital for x-values, you might just do:
    def partial_ai_capital(t: int) -> float:
        # In partial AI, you might not rely on a separate K(t),
        # or you can define some logic. For demonstration, let's do a constant K=1.
        return 1.0

    # (C) synergy_func, intangible_func remain the same as user-passed.

    # (D) We pass everything into compute_y_romer, telling it
    #     to interpret 'labor' as partial_ai_labor, and 'capital' as partial_ai_capital.
    #     This effectively does Y_new(t) = [ (LY + phi*KAI) ]^(1-alpha)* sum(x^alpha).

    return compute_y_romer(
        num_periods=num_periods,
        alpha=alpha,
        labor_func=partial_ai_labor,   # overrides
        capital_func=partial_ai_capital,
        # no special capital_updater unless you want it
        synergy_func=synergy_func,
        intangible_func=intangible_func,
        knowledge_updater=knowledge_updater,
        x_values_updater=x_values_updater,
        x_values_init=x_values_init,
        knowledge_init=knowledge_init,
        capital_init=capital_init,
        store_results=store_results
    )

def use_full_ai(
    num_periods: int,
    alpha: float,
    kai_func: Callable[[int], float],      # K_AI(t)
    phi_func: Callable[[int], float],      # phi(t)
    synergy_func: Callable[[int], float],
    intangible_func: Callable[[int], float],
    knowledge_updater: Optional[Callable] = None,
    x_values_updater: Optional[Callable] = None,
    x_values_init: np.ndarray = None,
    knowledge_init: float = 0.0,
    capital_init: float = 0.0,
    store_results: bool = True
) -> Dict[str, Any]:
    """
    Wrapper to run 'compute_y_romer' in a 'full AI' mode:
      Y_new(t) = [phi*KAI]^(1-alpha)* sum( x_i^alpha ), with LY ~ 0
    """
    # (A) define a labor_func returning 0 (or near-zero).
    def full_ai_labor(t: int) -> float:
        return 0.0  # effectively no human labor

    # (B) define a capital_func = 1.0 or something, unless you want
    #     a real capital approach. Often full AI invests in KAI, so
    #     you might or might not want a separate aggregator for x_i.
    def full_ai_capital(t: int) -> float:
        return 1.0  # or some user logic

    # (C) synergy, intangible as user-provided.

    # (D) now compute_y_romer sees LY(t)=0 => aggregator is
    #     [0]^(1-alpha)* sum(...) = 0, which is not what we want. Actually, to do
    #     Y = [phi*KAI]^(1-alpha)* sum(x^alpha), we can do a trick:
    #     treat 'labor' as phi*KAI so aggregator is [phi*KAI]^(1-alpha).
    #     But we also want the same phi*KAI in x-values?? Typically, you'd do:
    def full_ai_labor(t: int) -> float:
        # aggregator sees "LY(t)" => we return phi(t)*KAI(t)
        phi_t = phi_func(t)
        k_t = kai_func(t)
        return phi_t * k_t  # aggregator => [phi*KAI]^(1-alpha)

    return compute_y_romer(
        num_periods=num_periods,
        alpha=alpha,
        labor_func=full_ai_labor,
        capital_func=full_ai_capital,
        synergy_func=synergy_func,
        intangible_func=intangible_func,
        knowledge_updater=knowledge_updater,
        x_values_updater=x_values_updater,
        x_values_init=x_values_init,
        knowledge_init=knowledge_init,
        capital_init=capital_init,
        store_results=store_results
    )

def compute_partial_knowledge_AI(
    num_periods: int,
    delta: float,
    theta: float,
    eta: float,
    dt: float,
    # Functions to retrieve each time step:
    A_init: float,  # initial knowledge A(0)
    LA_func: Callable[[int], float],      # returns L_A(t) >= 0
    KAI_R_func: Callable[[int], float],   # returns K_AI,R(t) >= 0
    gamma_func: Callable[[int], float],   # returns gamma(t) >= 0
    synergy_func: Callable[[int], float], # synergy(t) >= 0
    intangible_func: Callable[[int], float], # intangible(t) >= 0
    # Whether to store step-by-step logs
    store_results: bool = True
) -> Dict[str, Any]:
    """
    Runs a multi-period simulation of partial-AI knowledge growth, as described by:
        dA/dt = delta * A^theta * [ L_A(t) + gamma(t)*K_AI,R(t) ]^eta,
    using a simple Euler integration with step size dt.

    We also incorporate synergy(t), intangible(t) placeholders so that L_A(t), gamma(t),
    or K_AI,R(t) can be shaped by synergy/intangible in user-supplied logic.

    For each step t in [0..num_periods-1]:
      1) synergy = synergy_func(t), intangible = intangible_func(t)
      2) LA = LA_func(t), KAI_R = KAI_R_func(t), gamma = gamma_func(t)
      3) A_dot = delta * A_current^theta * [ LA + gamma*KAI_R ]^eta
      4) A_next = A_current + dt*A_dot
      5) Validate no negative knowledge. Optionally store step results.

    :param num_periods: int >=1
        Number of discrete steps to simulate.
    :param delta: float >=0
        R&D productivity factor in partial-AI knowledge equation.
    :param theta: float >=0
        Exponent on A(t).
    :param eta: float >=0
        Exponent on [LA + gamma*KAI_R].
    :param dt: float >0
        Time-step for Euler integration.
    :param A_init: float >=0
        Initial knowledge stock A(0).
    :param LA_func: function(t)->float
        Yields L_A(t) >=0 each step (R&D labor).
    :param KAI_R_func: function(t)->float
        Yields K_AI,R(t) >=0 each step (AI capital allocated to R&D).
    :param gamma_func: function(t)->float
        Yields gamma(t) >=0, synergy for partial AI in R&D.
    :param synergy_func: function(t)->float
        synergy(t) >=0. Not directly in the formula, but can shape LA_func or gamma_func.
    :param intangible_func: function(t)->float
        intangible(t) >=0. Similarly might shape user logic if you want more advanced expansions.
    :param store_results: bool
        If True, store step logs. If False, only final state is returned.

    :return: dict with:
        {
          "outputs": step logs if store_results=True,
          "final_knowledge": final A(t),
          "final_synergy": synergy(t) from last step,
          "final_intangible": intangible(t) from last step,
          "final_LA": LA(t) from last step,
          "final_KAI_R": KAI_R(t) from last step,
          "final_gamma": gamma(t) from last step
        }
    :raises ValueError:
        If negative or invalid inputs (e.g., delta<0, dt<=0, synergy<0, intangible<0,
        LA<0, gamma<0, KAI_R<0, or if the integration yields negative A_next).
    """

    # ---------------------------------------------------------
    # 0) Validate Basic Parameters
    # ---------------------------------------------------------
    if not isinstance(num_periods, int) or num_periods < 1:
        raise ValueError(f"num_periods must be a positive int. Got {num_periods}")
    if delta < 0:
        raise ValueError(f"delta must be >=0, got {delta}")
    if theta < 0:
        raise ValueError(f"theta must be >=0, got {theta}")
    if eta < 0:
        raise ValueError(f"eta must be >=0, got {eta}")
    if dt <= 0:
        raise ValueError(f"dt must be >0, got {dt}")
    if A_init < 0:
        raise ValueError(f"A_init must be >=0, got {A_init}")

    # We'll check synergy_func(0), intangible_func(0), etc. just to confirm no negative
    synergy_0 = synergy_func(0)
    intangible_0 = intangible_func(0)
    if synergy_0 < 0:
        raise ValueError(f"synergy_func(0) returned negative synergy.")
    if intangible_0 < 0:
        raise ValueError(f"intangible_func(0) returned negative intangible.")

    # ---------------------------------------------------------
    # 1) Initialize State
    # ---------------------------------------------------------
    A_current = A_init
    synergy_current = synergy_0
    intangible_current = intangible_0

    results = []
    LA_current = None
    KAI_R_current = None
    gamma_current = None

    # ---------------------------------------------------------
    # 2) Main Loop
    # ---------------------------------------------------------
    for t in range(num_periods):
        # 2a) synergy, intangible, LA, KAI_R, gamma
        synergy_t = synergy_func(t)
        intangible_t = intangible_func(t)
        LA_t = LA_func(t)
        KAI_R_t = KAI_R_func(t)
        gamma_t = gamma_func(t)

        # Basic validations
        if synergy_t < 0:
            raise ValueError(f"synergy_func returned negative synergy at step {t}: {synergy_t}")
        if intangible_t < 0:
            raise ValueError(f"intangible_func returned negative intangible at step {t}: {intangible_t}")
        if LA_t < 0:
            raise ValueError(f"LA_func returned negative L_A at step {t}: {LA_t}")
        if KAI_R_t < 0:
            raise ValueError(f"KAI_R_func returned negative KAI_R at step {t}: {KAI_R_t}")
        if gamma_t < 0:
            raise ValueError(f"gamma_func returned negative gamma at step {t}: {gamma_t}")

        # 2b) Compute derivative: A_dot = delta * A^theta * [LA + gamma*KAI_R]^eta
        A_dot = compute_A_dot_partial_AI_single_step(
            A_current, LA_t, KAI_R_t, delta, theta, gamma_t, eta
        )

        # 2c) Integrate: A_next = A_current + dt*A_dot
        A_next = A_current + dt*A_dot
        if A_next < 0:
            raise ValueError(f"Euler integration gave negative A_next at step {t}: {A_next}")

        # 2d) Optionally store results
        if store_results:
            results.append({
                "t": t,
                "A_current": A_current,
                "A_dot": A_dot,
                "LA": LA_t,
                "KAI_R": KAI_R_t,
                "gamma": gamma_t,
                "synergy": synergy_t,
                "intangible": intangible_t,
                "A_next": A_next
            })

        # 2e) Commit updates
        A_current = A_next
        synergy_current = synergy_t
        intangible_current = intangible_t
        LA_current = LA_t
        KAI_R_current = KAI_R_t
        gamma_current = gamma_t

    # ---------------------------------------------------------
    # 3) Final Output
    # ---------------------------------------------------------
    return {
        "outputs": results if store_results else None,
        "final_knowledge": A_current,
        "final_synergy": synergy_current,
        "final_intangible": intangible_current,
        "final_LA": LA_current,
        "final_KAI_R": KAI_R_current,
        "final_gamma": gamma_current
    }

def compute_A_dot_partial_AI_single_step(
    A: float,
    LA: float,
    KAI_R: float,
    delta: float,
    theta: float,
    gamma_val: float,
    eta: float
) -> float:
    """
    Single-step partial AI knowledge derivative:
      A_dot = delta * A^theta * [LA + gamma*KAI_R]^eta

    :param A: float >=0, knowledge stock
    :param LA: float >=0, R&D labor
    :param KAI_R: float >=0, AI capital allocated to R&D
    :param delta: float >=0, productivity factor
    :param theta: float >=0, exponent on A
    :param gamma_val: float >=0, synergy for AI in R&D
    :param eta: float >=0, exponent on [LA + gamma*KAI_R]
    :return: float, dA/dt at this instant
    :raises ValueError:
        If any input is negative or inconsistent.
    """
    # Validate inputs
    if A < 0:
        raise ValueError(f"A must be >=0, got {A}")
    if LA < 0:
        raise ValueError(f"LA must be >=0, got {LA}")
    if KAI_R < 0:
        raise ValueError(f"KAI_R must be >=0, got {KAI_R}")
    if delta < 0:
        raise ValueError(f"delta must be >=0, got {delta}")
    if theta < 0:
        raise ValueError(f"theta must be >=0, got {theta}")
    if gamma_val < 0:
        raise ValueError(f"gamma_val must be >=0, got {gamma_val}")
    if eta < 0:
        raise ValueError(f"eta must be >=0, got {eta}")

    base_term = LA + gamma_val*KAI_R
    return delta * (A ** theta) * (base_term ** eta)

def compute_full_knowledge_AI(
    num_periods: int,
    dt: float,
    delta: float,
    theta: float,
    eta: float,
    A_init: float,
    # synergy, intangible, KAI_R, gamma as callables:
    synergy_func: Callable[[int], float],
    intangible_func: Callable[[int], float],
    KAI_R_func: Callable[[int], float],
    gamma_func: Callable[[int], float],
    # optional storing of step results
    store_results: bool = True
) -> Dict[str, Any]:
    """
    Runs a multi-period simulation of "full-AI" knowledge growth, 
    as described by the differential equation:
        dA/dt = delta * [A(t)]^theta * [gamma(t)*KAI_R(t)]^eta,
    where L_A(t) ~ 0. We use a simple Euler integration with step size dt.

    For each time step t in [0..num_periods-1]:
      1) synergy_t = synergy_func(t) >=0
      2) intangible_t = intangible_func(t) >=0
      3) KAI_R_t = KAI_R_func(t) >=0
      4) gamma_t = gamma_func(t) >=0
      5) A_dot = compute_A_dot_full_ai_single_step(A_current, KAI_R_t, delta, theta, gamma_t, eta)
      6) A_next = A_current + dt*A_dot
      7) Validate that A_next >=0
      8) Optionally store step results
      9) Move to t+1

    synergy_func, intangible_func might not directly appear in the formula,
    but we provide them to unify partial/full AI frameworks or advanced expansions 
    if user logic ties synergy/intangible to KAI_R(t) or gamma(t).

    :param num_periods: int >=1
        Number of discrete time steps.
    :param dt: float >0
        The Euler step size for knowledge updates.
    :param delta: float >=0
        Productivity factor in full AI knowledge equation.
    :param theta: float >=0
        Exponent on A(t).
    :param eta: float >=0
        Exponent on [gamma*KAI_R].
    :param A_init: float >=0
        Initial knowledge stock A(0).
    :param synergy_func: function(t)->float
        synergy(t) >= 0 each step. 
    :param intangible_func: function(t)->float
        intangible(t) >=0 each step.
    :param KAI_R_func: function(t)->float
        KAI_R(t) >=0 each step (AI capital allocated for R&D).
    :param gamma_func: function(t)->float
        gamma(t) >=0 synergy coefficient for full AI knowledge. 
    :param store_results: bool
        If True, store step results in "outputs." 
        If False, only final state is returned.

    :return: Dict with:
        {
          "outputs": [list of step logs if store_results=True],
          "final_knowledge": final A(t),
          "final_synergy": synergy from last step,
          "final_intangible": intangible from last step,
          "final_KAI_R": KAI_R from last step,
          "final_gamma": gamma from last step
        }
    :raises ValueError:
        if negative or invalid inputs (dt<=0, synergy<0, intangible<0, KAI_R<0, gamma<0)
        or if A_next <0 after integration, etc.
    """

    # ----------------------------------------------------------------
    # 0) Validate Inputs
    # ----------------------------------------------------------------
    if not isinstance(num_periods, int) or num_periods < 1:
        raise ValueError(f"num_periods must be a positive int. Got {num_periods}")
    if dt <= 0:
        raise ValueError(f"dt must be >0, got {dt}")
    if delta < 0:
        raise ValueError(f"delta must be >=0, got {delta}")
    if theta < 0:
        raise ValueError(f"theta must be >=0, got {theta}")
    if eta < 0:
        raise ValueError(f"eta must be >=0, got {eta}")
    if A_init < 0:
        raise ValueError(f"A_init must be >=0, got {A_init}")

    # synergy, intangible test at t=0
    synergy_0 = synergy_func(0)
    intangible_0 = intangible_func(0)
    if synergy_0 < 0:
        raise ValueError("synergy_func(0) returned negative synergy.")
    if intangible_0 < 0:
        raise ValueError("intangible_func(0) returned negative intangible.")

    # ----------------------------------------------------------------
    # 1) Initialize State
    # ----------------------------------------------------------------
    A_current = A_init
    synergy_current = synergy_0
    intangible_current = intangible_0
    KAI_R_current = KAI_R_func(0)
    gamma_current = gamma_func(0)

    if KAI_R_current < 0:
        raise ValueError(f"KAI_R_func(0) returned negative {KAI_R_current}")
    if gamma_current < 0:
        raise ValueError(f"gamma_func(0) returned negative {gamma_current}")

    results = []

    # ----------------------------------------------------------------
    # 2) Main Loop
    # ----------------------------------------------------------------
    for t in range(num_periods):
        # 2a) synergy, intangible, KAI_R, gamma
        synergy_t = synergy_func(t)
        intangible_t = intangible_func(t)
        KAI_R_t = KAI_R_func(t)
        gamma_t = gamma_func(t)

        if synergy_t < 0:
            raise ValueError(f"synergy_func returned negative synergy({t}): {synergy_t}")
        if intangible_t < 0:
            raise ValueError(f"intangible_func returned negative intangible({t}): {intangible_t}")
        if KAI_R_t < 0:
            raise ValueError(f"KAI_R_func returned negative KAI_R({t}): {KAI_R_t}")
        if gamma_t < 0:
            raise ValueError(f"gamma_func returned negative gamma({t}): {gamma_t}")

        # 2b) Compute derivative
        A_dot = compute_A_dot_full_AI_single_step(
            A_current, KAI_R_t, delta, theta, gamma_t, eta
        )

        # 2c) Euler Step: A_next = A_current + dt*A_dot
        A_next = A_current + dt*A_dot
        if A_next < 0:
            raise ValueError(f"Integration gave negative knowledge at step {t}: {A_next}")

        # 2d) Possibly Store Results
        if store_results:
            results.append({
                "t": t,
                "A_current": A_current,
                "A_dot": A_dot,
                "synergy": synergy_t,
                "intangible": intangible_t,
                "KAI_R": KAI_R_t,
                "gamma": gamma_t,
                "A_next": A_next
            })

        # 2e) Commit updates
        A_current = A_next
        synergy_current = synergy_t
        intangible_current = intangible_t
        KAI_R_current = KAI_R_t
        gamma_current = gamma_t

    # ----------------------------------------------------------------
    # 3) Final Output
    # ----------------------------------------------------------------
    return {
        "outputs": results if store_results else None,
        "final_knowledge": A_current,
        "final_synergy": synergy_current,
        "final_intangible": intangible_current,
        "final_KAI_R": KAI_R_current,
        "final_gamma": gamma_current
    }


def compute_A_dot_full_AI_single_step(
    A: float,
    KAI_R: float,
    delta: float,
    theta: float,
    gamma_val: float,
    eta: float
) -> float:
    """
    Single-step derivative for "full AI" knowledge:
      A_dot = delta * [A]^theta * [gamma*KAI_R]^eta,
    with L_A(t) ~ 0 (human labor not used in knowledge production).

    :param A: float >=0, knowledge stock
    :param KAI_R: float >=0, AI capital allocated to knowledge production
    :param delta: float >=0, productivity
    :param theta: float >=0, exponent on A
    :param gamma_val: float >=0, synergy for AI in knowledge
    :param eta: float >=0, exponent on [gamma*KAI_R]
    :return: float, derivative dA/dt at this moment
    :raises ValueError:
        if any input is negative or invalid
    """
    if A < 0:
        raise ValueError(f"A must be >=0, got {A}")
    if KAI_R < 0:
        raise ValueError(f"KAI_R must be >=0, got {KAI_R}")
    if delta < 0:
        raise ValueError(f"delta must be >=0, got {delta}")
    if theta < 0:
        raise ValueError(f"theta must be >=0, got {theta}")
    if gamma_val < 0:
        raise ValueError(f"gamma_val must be >=0, got {gamma_val}")
    if eta < 0:
        raise ValueError(f"eta must be >=0, got {eta}")

    return delta * (A ** theta) * ((gamma_val * KAI_R) ** eta)

def compute_org_intang_capital(
    num_periods: int,
    delta_oc: float,
    g_sga: float,
    # NOTE: If you want the "classic" 0.15 depreciation rate, set delta_oc=0.15
    # and if your typical real growth rate is 0.10, set g_sga=0.10

    # -- 1) Functions for retrieving SGA(t) and CPI(t) each step --
    sga_func: Callable[[int], float],
    cpi_func: Callable[[int], float],

    # -- 2) synergy, knowledge placeholders (optional but consistent with your approach) --
    synergy_func: Callable[[int], float] = lambda t: 0.0,
    knowledge_func: Callable[[int], float] = lambda t: 0.0,

    # -- 3) Logistic Acceleration parameters (kappa, U_bar) if you want intangible synergy --
    # If you do NOT want logistic synergy, pass in lambdas returning 0 or a constant.
    kappa_func: Callable[[int, float, float, float], float] = lambda t, syn, kn, oc_t: 0.0,
    U_bar_func: Callable[[int, float, float, float], float] = lambda t, syn, kn, oc_t: 0.0,

    # -- 4) Optional "additional invests" for intangible outside SGA/CPI,
    #       if you'd like to combine your previous intangible_invest_func logic. --
    extra_invest_func: Callable[[int, float, float, float], float] = lambda t, syn, kn, oc_t: 0.0,

    # -- 5) We store or skip logs. You can set store_results=False for performance. --
    store_results: bool = True,

    # -- 6) Initial conditions --
    # We'll unify "OC(0)" and "SGA(1)/(g + delta_oc)" logic, but you must supply SGA(1) if you'd like a dynamic start.
    # If you prefer you can pass a custom numeric for oc_init. We allow either approach.
    oc_init: Optional[float] = None,
    # The time index for your first "true" SGA data. If your data starts at t=0,
    # but your "initial stock" references SGA(1), you can set sga_initial_index=1.
    sga_initial_index: int = 1
) -> Dict[str, Any]:
    """
    Compute the "Organization Capital" (OC) or intangible capital stock across multiple periods,
    blending:
      (A) The perpetual-inventory approach from Eisfeldt & Papanikolaou (2013):
            OC_{t+1} = (1 - delta_oc) * OC_t + (SGA(t+1)/CPI(t+1))
          with an initial condition:
            OC(0) = SGA( first_observed_period ) / (g_sga + delta_oc)
      (B) An optional logistic acceleration factor (like your intangible logistic approach):
            new_invest = [SGA(t+1)/CPI(t+1) + extra_invest_func(...)]
                         * [1 / (1 + exp(-kappa(t)*(OC_t - U_bar(t))))]
          and then also applying depreciation on the existing OC_t.

    For each discrete time step t in [0..num_periods-1], we do:

      1) synergy_t = synergy_func(t)
      2) knowledge_t = knowledge_func(t)
      3) sga_t = sga_func(t)
      4) cpi_t = cpi_func(t)
      5) invests_sga = sga_t / cpi_t
      6) invests_extra = extra_invest_func(t, synergy_t, knowledge_t, OC_current)
      7) total_invests = invests_sga + invests_extra
      8) logistic_factor = 1/(1 + exp(-kappa(t, synergy_t, knowledge_t, OC_current)
                                    * (OC_current - U_bar(t, synergy_t, knowledge_t, OC_current))))
      9) invest_increment = total_invests * logistic_factor
      10) depreciation = (1 - delta_oc)*OC_current
      11) OC_{t+1} = depreciation + invest_increment

    NOTE: If you want a strict replication of the basic formula from the SAS code
    (without logistic synergy or extra invests), set:
        kappa_func(...) => 0.0
        extra_invest_func(...) => 0.0
        (thus logistic_factor => 1/(1 + exp(0)) => ~0.5, so you might also pass a big kappa=0 => factor=1.0
        or you can define a function returning kappa=1 but always U_bar= -9999 so factor ~1.0
    In other words, you have full freedom to incorporate synergy/logistic or not.

    :param num_periods: int >= 1
        Number of discrete time steps you want to simulate.
        Typically you'd set it to the # of years or data points you have.
    :param delta_oc: float in [0,1]
        Depreciation rate for organization capital. E.g., 0.15.
    :param g_sga: float >= 0
        Average real growth rate for SG&A, e.g. 0.10. Used only if oc_init is None.
    :param sga_func: function(t)->float
        Returns the SGA expense at time t.
    :param cpi_func: function(t)->float
        Returns the CPI index at time t (nonzero).
    :param synergy_func: function(t)->float
        synergy(t) >= 0. If not needed, can return 0.0.
    :param knowledge_func: function(t)->float
        knowledge(t) >= 0. If not needed, can return 0.0.
    :param kappa_func: function(t, synergy, knowledge, oc_current)->float
        If >0, shapes how intangible invests saturate as OC rises past U_bar(t).
        If you do not want the logistic effect, return 0 or pass a trivial lambda.
    :param U_bar_func: function(t, synergy, knowledge, oc_current)->float
        The intangible threshold. If not used, can return 0 or a big negative to ensure factor ~1.0.
    :param extra_invest_func: function(t, synergy, knowledge, oc_current)->float
        Additional intangible invests you want to inject each period outside the SGA(t)/CPI(t) logic.
        Defaults to 0.0 if not used.
    :param store_results: bool
        If True, store each iteration's states in 'outputs'.
        If False, returns only final results (for performance).
    :param oc_init: optional float
        If you want to override the initial stock directly. Otherwise we do
          OC(0) = SGA(sga_initial_index)/(g_sga + delta_oc)
        using your SGA data from the chosen time index. 
    :param sga_initial_index: int
        The time index to pick from sga_func for the initial stock if oc_init is None.
        E.g. if your first real SGA is at t=1, set sga_initial_index=1.

    :return: A dictionary with final and/or step-by-step data:
        {
          "outputs": [...],                 # each step if store_results=True
          "final_OC": float,               # final intangible/organization capital
          "final_synergy": float,          # synergy at last iteration
          "final_knowledge": float,        # knowledge at last iteration
          "final_invest_sga": float,       # last SGA/CPI invests
          "final_invest_extra": float,     # last extra invests
          "final_kappa": float,           # last slope
          "final_U_bar": float            # last threshold
        }
    :raises ValueError:
        If any function returns negative or invalid values, or if the final results produce negative capital, etc.
    """
    # ----------------------------------------------------------------
    # (A) Validate Basic Inputs
    # ----------------------------------------------------------------
    if not isinstance(num_periods, int) or num_periods < 1:
        raise ValueError(f"num_periods must be a positive integer, got {num_periods}")
    if not (0.0 <= delta_oc < 1.0):
        raise ValueError(f"delta_oc must be in [0,1). Got {delta_oc}")
    if g_sga < 0:
        raise ValueError(f"g_sga must be >= 0. Got {g_sga}")

    # ----------------------------------------------------------------
    # (B) Initialize the capital stock
    # ----------------------------------------------------------------
    # If user provided oc_init, we trust it. Otherwise, we do the formula:
    #   OC(0) = SGA(sga_initial_index) / (g_sga + delta_oc)
    if oc_init is not None:
        oc_current = oc_init
    else:
        # pull the sga at the special index
        sga_initial = sga_func(sga_initial_index)
        if sga_initial < 0:
            raise ValueError(f"sga_func({sga_initial_index}) returned negative SGA: {sga_initial}")
        denom = (g_sga + delta_oc)
        if denom <= 0:
            raise ValueError("g_sga + delta_oc must be > 0 to compute initial stock.")
        oc_current = sga_initial / denom

    # We do a quick synergy/knowledge check at t=0
    syn_0 = synergy_func(0)
    kn_0 = knowledge_func(0)
    if syn_0 < 0:
        raise ValueError(f"synergy_func(0) returned negative {syn_0}")
    if kn_0 < 0:
        raise ValueError(f"knowledge_func(0) returned negative {kn_0}")

    # For storing iteration logs
    results = []
    final_invest_sga = None
    final_invest_extra = None
    final_kappa = None
    final_U_bar = None

    # ----------------------------------------------------------------
    # (C) Main Loop
    # ----------------------------------------------------------------
    for t in range(num_periods):
        synergy_t = synergy_func(t)
        knowledge_t = knowledge_func(t)
        if synergy_t < 0:
            raise ValueError(f"synergy_func returned negative synergy at step {t}: {synergy_t}")
        if knowledge_t < 0:
            raise ValueError(f"knowledge_func returned negative knowledge at step {t}: {knowledge_t}")

        sga_t = sga_func(t)
        if sga_t < 0:
            raise ValueError(f"sga_func returned negative SGA at step {t}: {sga_t}")
        cpi_t = cpi_func(t)
        if cpi_t <= 0:
            raise ValueError(f"cpi_func returned non-positive CPI at step {t}: {cpi_t}")

        # 1) invests_sga = SGA(t)/CPI(t)
        invests_sga = sga_t / cpi_t
        # 2) invests_extra = optional
        invests_extra = extra_invest_func(t, synergy_t, knowledge_t, oc_current)
        if invests_extra < 0:
            raise ValueError(f"extra_invest_func returned negative invests at step {t}: {invests_extra}")

        total_invests = invests_sga + invests_extra

        # 3) logistic factor
        kappa_t = kappa_func(t, synergy_t, knowledge_t, oc_current)
        if kappa_t < 0:
            raise ValueError(f"kappa_func returned negative kappa at step {t}: {kappa_t}")
        U_bar_t = U_bar_func(t, synergy_t, knowledge_t, oc_current)
        # We do not forcibly require U_bar_t>0. It's user logic. 

        denom = 1.0 + np.exp(-kappa_t*(oc_current - U_bar_t))
        logistic_factor = 1.0/denom  # in [0,1] if kappa>=0

        # 4) Effective invests: total_invests * logistic_factor
        invest_increment = total_invests * logistic_factor

        # 5) Depreciation portion: (1 - delta_oc)*oc_current
        #    Then we add the new invests to get oc_next
        oc_next = (1.0 - delta_oc)*oc_current + invest_increment
        if oc_next < 0:
            raise ValueError(f"Calculated negative capital at step {t}: {oc_next}")

        # optionally store
        if store_results:
            results.append({
                "t": t,
                "OC_current": oc_current,
                "synergy": synergy_t,
                "knowledge": knowledge_t,
                "SGA": sga_t,
                "CPI": cpi_t,
                "invest_sga": invests_sga,
                "invest_extra": invests_extra,
                "kappa": kappa_t,
                "U_bar": U_bar_t,
                "logistic_factor": logistic_factor,
                "OC_next": oc_next
            })

        # commit
        oc_current = oc_next
        final_invest_sga = invests_sga
        final_invest_extra = invests_extra
        final_kappa = kappa_t
        final_U_bar = U_bar_t

    # ----------------------------------------------------------------
    # (D) Return final state
    # ----------------------------------------------------------------
    return {
        "outputs": results if store_results else None,
        "final_OC": oc_current,
        "final_synergy": synergy_t,
        "final_knowledge": knowledge_t,
        "final_invest_sga": final_invest_sga,
        "final_invest_extra": final_invest_extra,
        "final_kappa": final_kappa,
        "final_U_bar": final_U_bar
    }

