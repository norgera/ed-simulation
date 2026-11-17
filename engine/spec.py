"""The process spec: which resources the process uses, where each is seized and released, and its capacity.

An event log shows WHAT happened. It cannot always show which resources were busy and for how long, so the engine
drafts a spec from the log and the user confirms or edits it (a small JSON file)."""
import json, math
import numpy as np, pandas as pd
from .discover import with_next

def concurrency_max(starts, ends):
    """Peak number of overlapping [start, end] intervals."""
    ev = np.concatenate([np.c_[starts, np.ones(len(starts))], np.c_[ends, -np.ones(len(ends))]])
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]   # at equal times, releases (-1) sort before acquires (+1)
    return int(np.cumsum(ev[:, 1]).max())

def releases(p):
    """A pool's release event(s) as a list (a spec may give one event or several, e.g. a chair freed by a move or a departure)."""
    r = p.get("release")
    return [] if not r else [r] if isinstance(r, str) else list(r)

def hold_intervals(log, acquire, release):
    """[t_acquire, t_release] for every case that has both events: first acquire, then the first release event after it.
    `release` is one event name or a list of them."""
    rel = [release] if isinstance(release, str) else list(release)
    a = log[log.activity == acquire].groupby("case_id").t.first().rename("a")
    r = log[log.activity.isin(rel)][["case_id", "t"]].merge(a, left_on="case_id", right_index=True)
    r = r[r.t >= r.a].groupby("case_id").t.min().rename("r")
    return pd.concat([a, r], axis=1).dropna()

def with_capacity(pools, capacity_set=None, capacity_delta=None):
    """Copy of the pool list with capacities changed. capacity_set {pool: n or [[start_hour, n], ...]} replaces the level;
    capacity_delta {pool: +/-n} adds to every shift."""
    out = []
    for p in pools:
        p = dict(p); v = (capacity_set or {}).get(p["name"])
        if isinstance(v, (int, float)): p["capacity"], p["shifts"] = int(v), None
        elif v: p["shifts"] = [list(x) for x in v]; p["capacity"] = max(c for _, c in v)
        d = (capacity_delta or {}).get(p["name"], 0)
        if d:
            p["capacity"] = max(0, int(p["capacity"]) + d)
            if p.get("shifts"): p["shifts"] = [[h, max(0, c + d)] for h, c in p["shifts"]]
        out.append(p)
    return out

def draft_spec(log, min_share=0.05):
    """Look for a resource label that appears on two consecutive events of a case: that pair is a hold."""
    d = with_next(log)
    d["res_next"] = d.groupby("case_id", sort=False)["resource"].shift(-1)
    start = log.groupby("case_id", sort=False).activity.first().value_counts().index[0]
    n_cases = log.case_id.nunique()
    notes, pools = [], []
    same = d[(d.resource.notna()) & (d.resource == d.res_next) & (d.activity != d.next_activity) & (d.activity != start)]
    for res, grp in same.groupby("resource"):
        pair = grp.groupby(["activity", "next_activity"]).size().sort_values(ascending=False)
        (a, b), n = pair.index[0], int(pair.iloc[0])
        if n < min_share * n_cases:
            continue
        iv = hold_intervals(log, a, b)
        pools.append(dict(name=str(res), acquire=a, release=b, capacity=concurrency_max(iv.a.values, iv.r.values),
                          shifts=None, priority_by_class=False, hold=None, note="drafted: capacity = peak overlap seen in the log"))
    seen = {p["name"] for p in pools}
    for res in log.resource.dropna().unique():
        if res not in seen and res not in set(log[log.activity == start].resource.dropna()):
            acts = log[log.resource == res].activity.value_counts()
            notes.append(f"Resource '{res}' appears on {list(acts.index)} but no start/end pair, so no pool was drafted. "
                         f"If it is a limited resource, add a pool with its acquire and release events (or an external hold).")
    return dict(start=start, class_col=None, pools=pools, notes=notes)

def load_spec(path):
    return json.load(open(path))

def save_spec(spec, path):
    json.dump(spec, open(path, "w"), indent=1)
