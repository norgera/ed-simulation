"""Process discovery: activities, directly-follows graph, start/terminal activities, and a Mermaid drawing."""
import pandas as pd

def with_next(log):
    """Add the next event within the same case: next_activity, next_t, gap (minutes)."""
    g = log.groupby("case_id", sort=False)
    d = log.copy()
    d["next_activity"] = g["activity"].shift(-1)
    d["next_t"] = g["t"].shift(-1)
    d["gap"] = d.next_t - d.t
    return d

def discover(log):
    d = with_next(log)
    n_cases = log.case_id.nunique()
    first = log.groupby("case_id", sort=False).activity.first().value_counts()
    last = log.groupby("case_id", sort=False).activity.last().value_counts()
    edges = d.dropna(subset=["next_activity"]).groupby(["activity", "next_activity"]).agg(
        n=("gap", "size"), mean_gap=("gap", "mean"), median_gap=("gap", "median")).reset_index()
    nodes = log.activity.value_counts()
    return dict(n_cases=int(n_cases), start=str(first.index[0]), starts=first.to_dict(),
                terminals={k: int(v) for k, v in last.items()}, nodes={k: int(v) for k, v in nodes.items()},
                edges=edges.to_dict("records"))

def mermaid(pm, min_share=0.01):
    """Mermaid flowchart of the discovered process; edges rarer than min_share of cases are hidden."""
    ids = {a: f"n{i}" for i, a in enumerate(pm["nodes"])}
    out = ["flowchart LR"]
    for a, n in pm["nodes"].items():
        shape = ('([', '])') if a == pm["start"] else (('[[', ']]') if a in pm["terminals"] else ('[', ']'))
        out.append(f'  {ids[a]}{shape[0]}"{a}<br/>{n}"{shape[1]}')
    cut = min_share * pm["n_cases"]
    for e in pm["edges"]:
        if e["n"] >= cut:
            out.append(f'  {ids[e["activity"]]} -->|"{e["n"]} · {e["mean_gap"]:.1f} min"| {ids[e["next_activity"]]}')
    return "\n".join(out)

# ---- merged process map: a start event and its end event become one process box ("Order start" + "Order end" -> "Order")
PAIRS = [("Lab", "Lab ordered/collected", "Lab result"), ("Imaging", "DI start", "DI result"),
         ("Consult", "Consult requested", "Consult completed"),
         ("Boarding", "Boarding start (admit request)", "Inpatient bed assigned")]   # ED pairs whose names do not end in start/end
SUFFIXES = [(" start", " end"), (" Start", " End"), ("_start", "_end"), (" begin", " end")]

def merge_rules(activities, pairs=PAIRS):
    """(process, start_event, end_event) for every '<X> start' / '<X> end' pair in the log, plus the explicit pairs."""
    acts = set(activities); rules, used = [], set()
    for a in sorted(acts):
        for s_, e_ in SUFFIXES:
            if a.endswith(s_) and a[: -len(s_)] + e_ in acts and a not in used:
                base = a[: -len(s_)]; rules.append((base, a, base + e_)); used |= {a, base + e_}
    rules += [r for r in pairs if r[1] in acts and r[2] in acts and r[1] not in used and r[2] not in used]
    return rules

def merged_map(log, rules=None):
    """Nodes: process (merged pair) or single event, with case count and median duration.
    Edges: which process follows which, with count and mean / median / p90 wait (next start - previous end, floored at 0
    because some steps run in parallel)."""
    rules = merge_rules(log.activity.unique()) if rules is None else rules
    start_of = {r[1]: r[0] for r in rules}; end_of = {r[2]: r[0] for r in rules}
    e = log[["case_id", "activity", "t"]].copy()
    e["proc"] = e.activity.map(start_of).fillna(e.activity.map(end_of)).fillna(e.activity)
    g = e.groupby(["case_id", "proc"], sort=False).agg(t_start=("t", "min"), t_end=("t", "max")).reset_index()
    g["dur"] = g.t_end - g.t_start
    g = g.sort_values(["case_id", "t_start", "t_end"], kind="mergesort")
    g["next"] = g.groupby("case_id", sort=False).proc.shift(-1)
    g["wait"] = (g.groupby("case_id", sort=False).t_start.shift(-1) - g.t_end).clip(lower=0)
    merged = {r[0] for r in rules}
    nodes = g.groupby("proc", sort=False).agg(n=("case_id", "size"), median_min=("dur", "median"), p90_min=("dur", lambda s: s.quantile(.9))).reset_index()
    nodes["merged"] = nodes.proc.isin(merged)
    edges = g.dropna(subset=["next"]).groupby(["proc", "next"], sort=False).agg(
        n=("wait", "size"), mean_wait=("wait", "mean"), median_wait=("wait", "median"), p90_wait=("wait", lambda s: s.quantile(.9))).reset_index()
    first = g.groupby("case_id", sort=False).proc.first().value_counts(); last = g.groupby("case_id", sort=False).proc.last().value_counts()
    return dict(n_cases=int(log.case_id.nunique()), rules=rules, start=str(first.index[0]), terminals={k: int(v) for k, v in last.items()},
                nodes=nodes.to_dict("records"), edges=edges.to_dict("records"))

def _fmt(m):
    return f"{m:.1f} min" if m < 60 else f"{m / 60:.1f} h"

def mermaid_merged(mm, min_share=0.01):
    """Mermaid flowchart: process boxes show how many cases and the median / p90 time in the process; arrows show the average
    wait before the next step (amber over 15 minutes, red over 60)."""
    ids = {n["proc"]: f"p{i}" for i, n in enumerate(mm["nodes"])}
    out = ["flowchart LR", "  classDef proc fill:#e8f1f8,stroke:#1f5f8b,color:#10212b",
           "  classDef event fill:#ffffff,stroke:#8a9895,color:#10212b", "  classDef term fill:#f2f2f2,stroke:#555,color:#10212b"]
    for n in mm["nodes"]:
        a = n["proc"]; i = ids[a]
        label = f'{a}<br/>{n["n"]:,}' + (f'<br/>median {_fmt(n["median_min"])} · p90 {_fmt(n["p90_min"])}' if n["merged"] else "")
        shape = ('([', '])') if a == mm["start"] else (('[[', ']]') if a in mm["terminals"] and not n["merged"] else ('[', ']'))
        out.append(f'  {i}{shape[0]}"{label}"{shape[1]}')
        out.append(f'  class {i} {"proc" if n["merged"] else ("term" if a in mm["terminals"] else "event")}')
    cut = min_share * mm["n_cases"]; k = 0
    for e in mm["edges"]:
        if e["n"] < cut: continue
        w = e["mean_wait"]
        out.append(f'  {ids[e["proc"]]} -->|"{e["n"]:,} · avg wait {_fmt(w)}"| {ids[e["next"]]}')
        col = "#c0392b" if w > 60 else "#d4880f" if w > 15 else "#5a6865"
        out.append(f"  linkStyle {k} stroke:{col},stroke-width:{1.5 + 3 * e['n'] / mm['n_cases']:.1f}px"); k += 1
    return "\n".join(out)
