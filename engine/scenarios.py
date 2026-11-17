"""What-if scenarios: change capacity, demand, step durations or external stays, rerun, and compare with the baseline.

A scenario is a dict: name, and any of
  capacity_set   {pool: n or [[start_hour, n], ...]}   set a resource level (a fixed number, or a shift pattern)
  capacity_delta {pool: +/-n}   add or remove units of a resource (shifts included)
  arrival_mult   number         scale the arrival rate
  hold_mult      {pool: f}      scale an external stay (e.g. ward length of stay)
  gap_mult       {activity: f}  scale the time taken to reach an activity (e.g. 0.75 on a result event = 25% faster)"""
import json
import numpy as np, pandas as pd
from .sim import simulate
from .kpis import summarize
from .spec import with_capacity

def spec_for(spec, kw):
    """The spec with a scenario's capacity changes, so log-based KPIs (utilization) use the scenario's capacity."""
    return {**spec, "pools": with_capacity(spec["pools"], kw.get("capacity_set"), kw.get("capacity_delta"))}

KEYS = ("capacity_set", "capacity_delta", "arrival_mult", "hold_mult", "gap_mult")

def run_scenarios(P, spec, scenarios, reps=3, days=60, warmup_days=10, target_util=0.8, log=print):
    terminals = set(P["terminals"]); rows = {}
    for sc in [dict(name="Baseline")] + list(scenarios):
        kw = {k: sc[k] for k in KEYS if k in sc}; vals, utils = [], []; sp = spec_for(spec, kw)
        for s in range(201, 201 + reps):
            lg, cls, util = simulate(P, days, warmup_days, s, **kw)
            r = summarize(lg, cls, sp, terminals).set_index(["kpi", "segment", "stat"]).value
            vals.append(r); utils.append(util)
        m = pd.concat(vals, axis=1).mean(axis=1); out = {}
        for (k, seg, stat), v in m.items():
            if seg == "All" or k == "outcome_share_pct":
                out[f"{k}|{seg}|{stat}"] = v
        for pool in utils[0]:
            out[f"sim_utilization_{pool}_pct"] = 100 * float(np.mean([u[pool] for u in utils]))
        rows[sc["name"]] = out; log(f"  {sc['name']}")
    df = pd.DataFrame(rows).T
    df.index.name = "scenario"
    df["pools_above_target"] = [", ".join(p for p in utils[0] if df.loc[n, f"sim_utilization_{p}_pct"] > 100 * target_util) for n in df.index]
    return df

def delta_table(df, cols):
    base = df.loc["Baseline", cols]
    d = (df[cols] / base - 1) * 100
    d.columns = [c + " (change %)" for c in cols]
    return d

def load_scenarios(path):
    return json.load(open(path))
