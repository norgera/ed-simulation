"""KPIs computed with one function, so the observed log and a simulated log are measured identically."""
import numpy as np, pandas as pd
from .discover import with_next
from .spec import hold_intervals, releases, concurrency_max

def _stats(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return {}
    return {"mean": x.mean(), "median": float(np.median(x)), "p90": float(np.quantile(x, .9))}

def avg_capacity(pool, T, hour0):
    if not pool.get("shifts"): return float(pool["capacity"])
    sh = sorted(pool["shifts"]); caps = []
    for m in np.arange(0, T, 15.0):
        h = (m / 60 + hour0) % 24; c = sh[-1][1]
        for s_, c_ in sh:
            if h >= s_: c = c_
        caps.append(c)
    return float(np.mean(caps))

def quit_ends(log, spec, terminals):
    """Endings that mean a case gave up: spec["abandon"], or every ending except the most common one (as in fit)."""
    if spec.get("abandon"): return set(spec["abandon"])
    ends = log.groupby("case_id", sort=False).activity.last().value_counts()
    return set(ends.index[1:]) & set(terminals)

def queue_intervals(d, pool, quits):
    """Waiting intervals for a pool: from the event before its acquire event to the acquire (served cases), and from that
    event to an abandonment ending for cases that gave up while waiting."""
    served = d[d.next_activity == pool["acquire"]]
    preds = set(served.activity)
    left = d[d.activity.isin(preds) & d.next_activity.isin(quits)]
    q = pd.concat([served, left])
    return q.t.values, q.next_t.values

def samples(log, classes, spec, terminals):
    """Per-case values behind the KPIs (for histograms): cycle time overall and by class, wait for each pool."""
    g = log.groupby("case_id", sort=False)
    ct = pd.DataFrame({"t0": g.t.first(), "t1": g.t.last(), "last": g.activity.last()})
    ct = ct[ct["last"].isin(terminals)]; cyc = ct.t1 - ct.t0
    out = {"cycle_time_min|All": cyc.values}
    if classes is not None:
        cl = classes.reindex(ct.index)
        for c in sorted(cl.dropna().unique()): out[f"cycle_time_min|class {c}"] = cyc[cl == c].values
    for o in ct["last"].unique(): out[f"cycle_time_min|ends {o}"] = cyc[ct["last"] == o].values
    d = with_next(log)
    for p in spec["pools"]:
        out[f"wait_for_{p['name']}_min|All"] = d[d.next_activity == p["acquire"]].gap.values
    return out

def summarize(log, classes, spec, terminals):
    rows = []
    def add(kpi, seg, stat, v): rows.append(dict(kpi=kpi, segment=seg, stat=stat, value=float(v)))
    g = log.groupby("case_id", sort=False)
    ct = pd.DataFrame({"t0": g.t.first(), "t1": g.t.last(), "last": g.activity.last()})
    ct = ct[ct["last"].isin(terminals)].copy(); ct["cycle"] = ct.t1 - ct.t0
    ct["cls"] = classes.reindex(ct.index) if classes is not None else "all"
    T = float(log.t.max()); days = T / 1440.0
    add("cases_per_day", "All", "value", len(ct) / days)
    for k, v in _stats(ct.cycle).items(): add("cycle_time_min", "All", k, v)
    if classes is not None:
        for c, sub in ct.groupby("cls"):
            for k, v in _stats(sub.cycle).items(): add("cycle_time_min", f"class {c}", k, v)
    for o, sub in ct.groupby("last"):
        add("outcome_share_pct", str(o), "value", 100 * len(sub) / len(ct))
        for k, v in _stats(sub.cycle).items(): add("cycle_time_min", f"ends {o}", k, v)
    d = with_next(log)
    hour0 = log.attrs["t0"].hour + log.attrs["t0"].minute / 60.0
    quits = quit_ends(log, spec, terminals)
    for p in spec["pools"]:
        w = d[d.next_activity == p["acquire"]].gap
        for k, v in _stats(w).items(): add(f"wait_for_{p['name']}_min", "All", k, v)
        s_, e_ = queue_intervals(d, p, quits)
        if len(s_):   # time-weighted average number waiting, and the longest queue seen
            add(f"queue_avg_{p['name']}", "All", "value", (np.clip(e_, 0, T) - np.clip(s_, 0, T)).sum() / T)
            add(f"queue_max_{p['name']}", "All", "value", concurrency_max(s_, e_))
        if releases(p):
            iv = hold_intervals(log, p["acquire"], releases(p))
            lo = np.clip(iv.a.values, 0, T); hi = np.clip(iv.r.values, 0, T)
            add(f"utilization_{p['name']}_pct", "All", "value", 100 * (hi - lo).sum() / (T * avg_capacity(p, T, hour0)))
    L = (np.minimum(ct.t1, T) - np.maximum(ct.t0, 0)).sum() / T
    add("little_L_over_lambdaW", "All", "value", L / ((len(ct) / T) * ct.cycle.mean()))
    return pd.DataFrame(rows)
