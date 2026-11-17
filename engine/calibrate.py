"""Calibrate a quantity the log cannot show, by bisection, so that one observable KPI of the simulation matches the log."""
import copy
import numpy as np
from .sim import simulate
from .kpis import summarize

def _kpi(P, spec, classes, kpi, stat, seeds, days, warmup=5):
    vals = []
    for s in seeds:
        log, cls, _ = simulate(P, days, warmup, s)
        r = summarize(log, cls, spec, set(P["terminals"]))
        vals.append(float(r[(r.kpi == kpi) & (r.stat == stat)].value.iloc[0]))
    return float(np.mean(vals))

def calibrate_visit_scale(P, spec, obs_summary, pool, kpi=None, stat="mean", lo=0.0, hi=0.9, iters=9, days=25, seeds=(11, 12), log=print):
    """Find the share of each hidden-work gap that is service by `pool`, so the simulated wait for `pool` matches the log."""
    kpi = kpi or f"wait_for_{pool}_min"
    target = float(obs_summary[(obs_summary.kpi == kpi) & (obs_summary.stat == stat)].value.iloc[0])
    P = copy.deepcopy(P)
    def run(k):
        for v in P["visits"]:
            if v["pool"] == pool: v["scale"] = k
        return _kpi(P, spec, None, kpi, stat, seeds, days)
    for _ in range(iters):
        mid = (lo + hi) / 2; v = run(mid)
        log(f"    scale={mid:.3f}: {kpi} {stat} = {v:.2f} (log: {target:.2f})")
        if v < target: lo = mid
        else: hi = mid
    k = (lo + hi) / 2; run(k)
    return P, k

def calibrate_hold_mean(P, spec, obs_summary, pool, kpi=None, stat="mean", iters=8, days=120, warmup=30, seeds=(11, 12, 13, 14), log=print):
    """Find the mean external stay (e.g. ward length of stay) so the simulated wait for `pool` (e.g. boarding) matches the log.
    Long warm-up and long runs: a nearly full ward takes weeks to settle, and the wait for it swings a lot between short runs."""
    kpi = kpi or f"wait_for_{pool}_min"
    target = float(obs_summary[(obs_summary.kpi == kpi) & (obs_summary.stat == stat)].value.iloc[0])
    P = copy.deepcopy(P); p = next(x for x in P["pools"] if x["name"] == pool)
    lo, hi = 0.6 * p["hold"]["mean"], 1.4 * p["hold"]["mean"]
    def run(m):
        p["hold"]["mean"] = m; return _kpi(P, spec, None, kpi, stat, seeds, days, warmup)
    for _ in range(iters):
        mid = (lo + hi) / 2; v = run(mid)
        log(f"    {pool} mean stay={mid:.0f} min: {kpi} {stat} = {v:.2f} (log: {target:.2f})")
        if v < target: lo = mid
        else: hi = mid
    p["hold"]["mean"] = (lo + hi) / 2
    return P, p["hold"]["mean"]

def unknowns(P):
    """What the log cannot show and the loop may tune: hidden-work shares (`visits`) and external stays (`hold` pools)."""
    return [("visit", v["pool"]) for v in P.get("visits", [])] + [("hold", p["name"]) for p in P["pools"] if p.get("hold")]

def validation_loop(P, spec, obs_log, obs_classes, tol=0.05, max_iter=4, reps=4, days=60, warmup_days=10, log=print):
    """Validate; if checks miss the tolerance, re-tune each unknown against the KPI it drives, and validate again.
    Stops when every check passes, when a round does not improve the count within tolerance, or after max_iter rounds.
    Returns the best params, its validation table and summary, the history of rounds, and the simulated logs of the best round."""
    from .validate import validate
    obs = summarize(obs_log, obs_classes, spec, set(P["terminals"]))
    best = None; history = []
    for it in range(max_iter + 1):
        val, summ, logs = validate(P, spec, obs_log, obs_classes, reps, days, warmup_days, tol=tol, keep_logs=True)
        n_ok = int((val.status == "PASS").sum()); n = summ["checks"]
        tuned = {f"{k}:{name}": (next(v["scale"] for v in P["visits"] if v["pool"] == name) if k == "visit"
                                 else next(p["hold"]["mean"] for p in P["pools"] if p["name"] == name)) for k, name in unknowns(P)}
        history.append(dict(round=it, passed=n_ok, checks=n, within_10pct=summ["within_10pct"], tuned=tuned))
        log(f"  round {it}: {n_ok}/{n} checks within {tol:.0%} (or within noise floor); tuned {tuned}")
        if best is None or n_ok > best[0]: best = (n_ok, copy.deepcopy(P), val, summ, logs)
        elif it > 0: log("  no improvement; stopping"); break
        if n_ok == n or it == max_iter or not unknowns(P): break
        for k, name in unknowns(P):   # re-tune with more replications each round so noise does not drive the result
            seeds = tuple(range(11, 13 + it))
            if k == "visit": P, _ = calibrate_visit_scale(P, spec, obs, name, seeds=seeds, log=log)
            else: P, _ = calibrate_hold_mean(P, spec, obs, name, seeds=tuple(range(11, 15 + it)), warmup=max(warmup_days, 30), log=log)
    _, P, val, summ, logs = best
    return P, val, summ, history, logs
