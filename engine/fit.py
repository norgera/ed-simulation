"""Fit the simulation inputs from the log: arrivals, class mix, transition probabilities, step durations, patience."""
import numpy as np, pandas as pd
from .discover import with_next

QGRID = np.unique(np.concatenate([np.linspace(0, .9, 91), np.linspace(.9, .99, 91), np.linspace(.99, 1, 101)]))

def quantiles(v):
    return np.quantile(np.asarray(v, float), QGRID).round(4).tolist()

def fit(log, classes, spec, min_class_n=50, min_leavers=20):
    d = with_next(log)
    d["cls"] = d.case_id.map(classes) if classes is not None else "all"
    d["cls"] = d.cls.fillna("unknown")
    nodes_out = set(d.dropna(subset=["next_activity"]).activity)
    terminals = set(log.activity) - nodes_out
    acq = {p["acquire"]: p for p in spec["pools"]}
    edges = d.dropna(subset=["next_activity"])

    # --- patience: a terminal edge out of a node that precedes a pool's acquire event is an abandonment.
    # Which endings mean "gave up" comes from spec["abandon"] when given; otherwise every ending except the most common
    # one (the normal end, e.g. a departure after a decision that also leads to a queue is a branch, not an abandonment).
    ends = log.groupby("case_id", sort=False).activity.last().value_counts()
    quit_ends = set(spec["abandon"]) if spec.get("abandon") else set(ends.index[1:]) & terminals
    patience, abandon = {}, {}
    for p in spec["pools"]:
        preds = set(edges[edges.next_activity == p["acquire"]].activity)
        stay = edges[(edges.next_activity == p["acquire"])]
        leave = edges[edges.activity.isin(preds) & edges.next_activity.isin(quit_ends)]
        if len(leave) >= min_leavers:
            patience[p["name"]] = float((leave.gap.sum() + stay.gap.sum()) / len(leave))
            for n_, grp in leave.groupby("activity"):
                abandon[n_] = grp.next_activity.value_counts().index[0]
    abandon_edges = {(n_, y) for n_, y in abandon.items()}

    # --- transitions and gap distributions, per class (pooled fallback when a class has few observations)
    def fit_edges(sub):
        out = {}
        for (a, b), g in sub.groupby(["activity", "next_activity"]):
            if (a, b) in abandon_edges and a in abandon and abandon[a] == b:
                continue
            out[(a, b)] = (len(g), None if b in acq else quantiles(g.gap.values))
        return out
    pooled = fit_edges(edges)
    trans = {}
    for c, sub in [("__all__", edges)] + list(edges.groupby("cls")):
        local = fit_edges(sub) if len(sub) >= min_class_n else {}
        # a class keeps its own probabilities, but borrows the pooled gap distribution for rarely seen edges
        tab = {}
        src = local if local else pooled
        for (a, b), (n, q) in src.items():
            if q is None and b not in acq:
                q = pooled[(a, b)][1]
            if b not in acq and (n < min_class_n) and (a, b) in pooled:
                q = pooled[(a, b)][1]
            tab.setdefault(a, []).append([b, n, q])
        for a, lst in tab.items():
            tot = sum(x[1] for x in lst)
            for x in lst:
                x[1] = x[1] / tot
        trans[c] = tab

    # --- arrivals: rate per minute by day-of-week and hour (hour only when under 4 weeks of data)
    start = spec["start"]
    arr = log[log.activity == start]
    t0 = log.attrs["t0"]
    span_h = int(np.ceil(log.t.max() / 60.0)) + 1
    slots = pd.date_range(t0.floor("h"), periods=span_h, freq="h")
    weeks_ok = span_h >= 28 * 24
    key = (lambda ts: (ts.dayofweek, ts.hour)) if weeks_ok else (lambda ts: (0, ts.hour))
    occ = pd.Series([key(s) for s in slots]).value_counts()
    cnt = pd.Series([key(s) for s in arr.ts]).value_counts()
    prof = np.zeros((7, 24))
    for k, o in occ.items():
        prof[k[0], k[1]] = cnt.get(k, 0) / o / 60.0
    if not weeks_ok:
        prof[:] = prof[0]
    mix = (pd.Series(classes).value_counts(normalize=True).to_dict() if classes is not None else {"all": 1.0})
    return dict(start=start, terminals=sorted(terminals), arrival=dict(profile=prof.round(6).tolist(), t0=str(t0)),
                class_mix=mix, trans=trans, abandon=abandon, patience=patience, pools=spec["pools"], visits=spec.get("visits", []),
                n_cases=int(log.case_id.nunique()), span_days=float(log.t.max() / 1440.0))
