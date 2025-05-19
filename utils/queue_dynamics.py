# utils/queue_dynamics.py
# ────────────────────────────────────────────────────────────────────────────
# FIFO idea-queue logic with latency bookkeeping.
#
#   • enqueue()      – push freshly triaged ideas into the queue
#   • update_queue() – pop ≤ ⌊Ψ_eff⌋ ideas for evaluation
#   • latency()      – helper to compute mean waiting-time KPI
#   • parquet_writer() – placeholder stub (full logic Phase E3)
# The queue is implemented as a collections.deque whose elements are
# (t_arrival, mu_post, var_post) tuples.  We keep the structure minimal
# so downstream modules (evaluation, knowledge updater, KPI derivations)
# can attach whatever extra fields they need without rewriting this core.
# ────────────────────────────────────────────────────────────────────────────
import logging
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils.triage_utils import bayes_posterior, triage_score
from utils.screening_utils import compute_threshold
from collections import namedtuple

# Convenience alias for the queue’s element type
IdeaTuple      = Tuple[int, float, float, float]          # (t_arrival, μ̂, σ̂², v₀)
IdeaEvalTuple  = Tuple[int, int, float, float, float]     # (t_arrival, t_eval, μ̂, σ̂², v₀)

LatencyStats = namedtuple("LatencyStats", ["count", "mean", "p95", "max", "std", "decay_loss"])    

def enqueue(
    q: Deque[IdeaTuple],
    t: int,
    mu_posts: Sequence[float],
    var_posts: Sequence[float],
    v0s: Sequence[float] | None = None,
) -> None:
    """
    Pushes a batch of triaged ideas (all from the same period *t*)
    into the FIFO queue *in place*.

    Parameters
    ----------
    q : deque
        The firm-specific FIFO queue (modified in place).
    t : int
        Current simulation period (non-negative).
    mu_posts, var_posts : sequence[float]
        Posterior mean and variance vectors *after* Bayesian update.
        Must be of equal length; their size is the batch size *N*.
    """
    if t < 0:
        raise ValueError("time index t must be non-negative")

    if len(mu_posts) != len(var_posts):
        raise ValueError("mu_posts and var_posts must have identical length")

    if any(v < 0 for v in var_posts):
        raise ValueError("posterior variances must be ≥ 0")

    if v0s is None:
        v0s = np.ones(len(mu_posts), dtype=float)
    if len(v0s) != len(mu_posts):
        raise ValueError("v0s must match the batch length of mu_posts")

    batch: List[IdeaTuple] = [
        (t, mu, var, v0) for mu, var, v0 in zip(mu_posts, var_posts, v0s)
    ]
    q.extend(batch)
    logging.debug("enqueue: %d ideas queued at t=%d", len(batch), t)


def enqueue_new_ideas(
    *,
    queue: Deque[IdeaTuple],
    n: int,
    t_arrival: int,
    mu_prior: float,
    tau_prior: float,
    sigma_noise: float,
    rng: np.random.Generator,
    v0_mu: float = 1.0,
    v0_sigma: float = 0.0,
    triage_params=None,           
) -> float:                                # ← lambda_weight arg removed
    """
    Convenience wrapper used by `ecb_firm_step`.

    • Simulates *n* true idea values v ~ N(μ_prior, τ²)
    • Adds noisy signals ε ~ N(0, σ²)
    • Computes posterior (μ̂, σ̂²)
    • Injects them into the queue with a log-normal base value v₀.
    """
    if n <= 0:
        return float("nan") 
    v_true  = rng.normal(loc=mu_prior, scale=tau_prior, size=n)
    signals = v_true + rng.normal(0.0, sigma_noise, size=n)

    mu_posts, var_posts, scores = [], [], []                            
    for s in signals:
        μ̂, σ̂2 = bayes_posterior(
            signal=s,
            mu_prior=mu_prior,
            tau_prior=tau_prior,
            sigma_noise=sigma_noise,
        )
        mu_posts.append(μ̂)
        var_posts.append(σ̂2)
        if triage_params is not None:                                   
            scores.append(                                               
                triage_score(mu_post=μ̂,                                 
                             var_post=σ̂2,                               
                             lambda_weight=triage_params.lambda_explore) 
            )                                                            

    triage_eff = np.nan                                                   
    if triage_params is not None:                                      
        cutoff = compute_threshold(                                     
            scores,                                                     
           rule = triage_params.threshold_rule,                       
            value= triage_params.threshold_value,                      
        )                                                                
        keep = [i for i,s in enumerate(scores) if s >= cutoff]          
        triage_eff = len(keep) / n                                      
        if not keep:                                               
            return triage_eff                                             
        mu_posts  = [mu_posts[i]  for i in keep]                       
        var_posts = [var_posts[i] for i in keep]                          
        # regenerate v0 only for survivors                               
        v0s = rng.lognormal(mean=v0_mu, sigma=v0_sigma, size=len(keep))  
    else:                                                                
        v0s = rng.lognormal(mean=v0_mu, sigma=v0_sigma, size=n)          

    enqueue(queue, t_arrival, mu_posts, var_posts, v0s)                  
    return triage_eff                                                   


def update_queue(
    q: Deque[IdeaTuple],
    psi_eff: float,
    *,
    t_eval: int | None = None,
    idea_log: List[IdeaEvalTuple] | None = None,
) -> List[IdeaEvalTuple] | List[IdeaTuple]:
    """
    Pops ≤⌊Ψ_eff⌋ ideas from *q* (FIFO) and returns them.

    If `t_eval` is given, every dequeued idea is augmented with that
    evaluation-time stamp and—optionally—appended to `idea_log`.  This is
    the Phase E3 pathway that enables latency / creativity-loss KPIs.

    Back-compat: callers that omit `t_eval` receive the pre-E3
    3-component tuples and no logging side-effects.

    Parameters
    ----------
    q         : deque[IdeaTuple]
    psi_eff   : float              – throughput (≥0)
    t_eval    : int | None         – current period (adds 2nd column)
    idea_log  : list | None        – global sink for idea-level records

    Returns
    -------
    list[IdeaEvalTuple] *or* list[IdeaTuple]
    """
    if psi_eff < 0:
        raise ValueError("psi_eff must be ≥ 0")
    n_served = min(len(q), int(psi_eff))
    popped   = [q.popleft() for _ in range(n_served)]

    # no evaluation time supplied
    if t_eval is None:
        logging.debug("update_queue: served %d ideas (legacy mode), queue_len=%d",
                      n_served, len(q))
        return popped                                    # type: ignore[return-value]

    # tack on evaluation timestamp
    served: List[IdeaEvalTuple] = [
        (t_arr, t_eval, mu, var, v0) for (t_arr, mu, var, v0) in popped
    ]
    if idea_log is not None:
        idea_log.extend(served)

    logging.debug("update_queue: served %d ideas @t=%d, queue_len=%d",
                  n_served, t_eval, len(q))
    return served

def service_queue_fifo(
    *,
    queue: Deque[IdeaTuple],
    capacity: float,
    t_now: int | None = None,
    rng: np.random.Generator | None = None,
    idea_log: List[IdeaEvalTuple] | None = None,
    eta_decay: float = 0.0,       
) -> Tuple[
        List[IdeaEvalTuple] | List[IdeaTuple],
        LatencyStats,
        float                                           
    ]:

    """
    Wrapper expected by `ecb_firm_step`. Supports *fractional* Ψ_eff:
    dequeue ⌊capacity⌋ ideas plus one extra with probability = frac(capacity).
    """
    if capacity < 0:
        raise ValueError("capacity must be ≥ 0")

    integral = int(np.floor(capacity))
    frac     = capacity - integral
    if frac > 0.0 and rng is None:
        rng = np.random.default_rng()

    served = update_queue(queue, integral, t_eval=t_now, idea_log=idea_log)
    if frac > 0.0 and rng.random() < frac:
        served.extend(update_queue(queue, 1, t_eval=t_now, idea_log=idea_log))

    waits = [t_now - t_arr for t_arr, *_ in served] if t_now is not None else []
    decay_loss = 0.0
    if eta_decay > 0.0 and t_now is not None:
        for t_arr, *_ , v0 in served:
            wait = t_now - t_arr
            decay_loss += v0 * (1.0 - np.exp(-eta_decay * wait))

        # ── additional loss for ideas that are *still waiting* ──
        for t_arr, *_ , v0 in queue:          # ← FIX: queue-deque, not “q”
            wait = t_now - t_arr               # they’ll wait one more period
            # incremental loss during *this* tick (Δw = 1)
            decay_loss += v0 * (
                np.exp(-eta_decay * wait) - np.exp(-eta_decay * (wait + 1))
            )
            
    stats = LatencyStats(
        count=len(waits),
        mean=np.mean(waits) if waits else np.nan,
        p95=np.percentile(waits, 95) if waits else np.nan,
        max=np.max(waits) if waits else np.nan,
        std=np.std(waits, ddof=0) if waits else np.nan,
        decay_loss=decay_loss, 
    )
    return served, stats, decay_loss

def latency(
    ideas: Iterable[IdeaTuple],
    t_now: int,
) -> float:
    """
    Computes *mean latency* (in periods) for a collection of ideas
    given the current period `t_now`.

    Parameters
    ----------
    ideas : iterable[IdeaTuple]
        Either the still-waiting queue **or** the batch just evaluated.
    t_now : int
        Current period.  Must be ≥ every arrival time in `ideas`.

    Returns
    -------
    float
        Average waiting time.  Returns 0.0 if the iterable is empty.
    """
    ideas = list(ideas)                # may consume generator
    if not ideas:
        return 0.0
    waits = [t_now - t_arr for t_arr, *_ in ideas]
    if any(w < 0 for w in waits):
        raise ValueError("t_now precedes some arrival times")
    return float(np.mean(waits))

def parquet_writer(
    ideas_log: List[IdeaEvalTuple],
    path: Path,
    debug: bool = True,
) -> None:
    """
    **Stub** – full writer spec comes in Phase E3 when we implement
    idea-level diagnostics.  For now we write a simple Parquet file
    if `debug` is True; otherwise we do nothing so production runs
    stay lightweight.

    The log row signature matches:
    (t_arrival, t_eval, μ̂, σ̂², v₀)

    Note
    ----
    This helper is side-effectful (I/O) but deliberately *idempotent*:
    we overwrite the Parquet file each call; callers can version paths
    with scenario_id / run_id if they need multiple logs.
    """
    if not debug or not ideas_log:
        return

    df = pd.DataFrame(
        ideas_log,
        columns=["t_arrival", "t_eval", "mu_post", "var_post", "v0"],
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    logging.info("parquet_writer: wrote %d rows to %s",
                 len(df), path.as_posix())


# Utility factory so callers can get a ready-to-use queue without
# importing collections.deque everywhere.
def new_queue() -> Deque[IdeaTuple]:
    """Returns an empty idea queue."""
    return deque()
