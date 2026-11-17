"""Run replications of the simulation, measure them with the same KPI code as the log, and compare."""
import numpy as np, pandas as pd
from .sim import simulate
from .kpis import summarize

def validate(P, spec, obs_log, obs_classes, reps=4, days=60, warmup_days=10, tol=0.05, seeds=None, keep_logs=False, **sim_kw):
    """Returns (table, summary); with keep_logs=True also the simulated (log, classes) pairs, for histograms."""
    from .scenarios import spec_for
    terminals = set(P["terminals"])
    obs = summarize(obs_log, obs_classes, spec, terminals)
    sims, utils, logs = [], [], []; sp = spec_for(spec, sim_kw)
    for s in (seeds or range(101, 101 + reps)):
        log, cls, util = simulate(P, days, warmup_days, s, **sim_kw)
        sims.append(summarize(log, cls, sp, terminals)); utils.append(util)
        if keep_logs: logs.append((log, cls))
    key = ["kpi", "segment", "stat"]
    S = pd.concat(sims).groupby(key).value.agg(sim_mean="mean", sim_sd="std").reset_index()
    out = obs.rename(columns={"value": "observed"}).merge(S, on=key, how="left")
    out["rel_err"] = np.where(out.observed.abs() > 1e-9, (out.sim_mean - out.observed) / out.observed.abs(), out.sim_mean - out.observed)
    def floor(kpi):   # differences smaller than this are not meaningful (a wait of 1 s vs 0.7 s is not a 40% error)
        return 0.1 if kpi.endswith("_min") else 0.5 if kpi.endswith("_pct") else 0.05 if kpi.startswith("queue_avg_") else 0.0
    def status(r):
        if pd.isna(r.sim_mean): return "MISSING"
        # the longest queue is an extreme value that grows with the length of the period, so it is shown, not scored
        if r.kpi == "little_L_over_lambdaW" or r.kpi.startswith("queue_max_"): return "INFO"
        if abs(r.rel_err) <= tol or abs(r.sim_mean - r.observed) <= floor(r.kpi): return "PASS"
        if abs(r.sim_mean - r.observed) <= 2 * (r.sim_sd if not pd.isna(r.sim_sd) else 0): return "NOISE"
        return "FAIL"
    out["status"] = out.apply(status, axis=1)
    chk = out[out.status != "INFO"]
    ok = lambda t: ((chk.rel_err.abs() <= t) | ((chk.sim_mean - chk.observed).abs() <= chk.kpi.map(floor))).sum()
    summary = {f"within_{int(t*100)}pct": int(ok(t)) for t in (0.05, 0.10, 0.20)}
    summary["checks"] = int(len(chk))
    summary["sim_pool_utilization"] = {k: float(np.mean([u[k] for u in utils])) for k in utils[0]}
    summary["tol"] = tol
    return (out, summary, logs) if keep_logs else (out, summary)
