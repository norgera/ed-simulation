"""KPI computation. The SAME code is applied to the observed log and to simulated logs (identical schema)."""
import numpy as np, pandas as pd

ACTS = ["Arrival","Triage start","Triage end","ED bed assigned","Physician initial assessment start",
        "Physician initial assessment end","Lab ordered/collected","Lab result","DI start","DI result",
        "Consult requested","Consult completed","Disposition decision","Boarding start (admit request)",
        "Inpatient bed assigned","Departure from ED","Left without being seen"]

def load(log_path, cases_path):
    ev = pd.read_csv(log_path, parse_dates=["timestamp"])
    cs = pd.read_csv(cases_path, parse_dates=["arrival_ts","departure_ts"])
    return ev, cs

def case_table(ev, cs):
    w = ev.pivot_table(index="case_id", columns="activity", values="timestamp", aggfunc="first")
    for a in ACTS:
        if a not in w: w[a] = pd.NaT
    ct = cs.set_index("case_id")[["ctas","outcome","admitted","arrival_ts","departure_ts"]].join(w)
    m = lambda a, b: (ct[b]-ct[a]).dt.total_seconds()/60
    ct["door_to_triage_min"] = m("Arrival","Triage start")
    ct["door_to_bed_min"]    = m("Arrival","ED bed assigned")
    ct["door_to_PIA_min"]    = m("Arrival","Physician initial assessment start")
    ct["boarding_min"]       = m("Boarding start (admit request)","Inpatient bed assigned")
    ct["LOS_min"]            = (ct.departure_ts-ct.arrival_ts).dt.total_seconds()/60
    done = ct[["Physician initial assessment end","Lab result","DI result","Consult completed"]].max(axis=1)
    ct["workup_done"] = done
    ct["decision_interval_min"] = (ct["Disposition decision"]-done).dt.total_seconds()/60
    ct["discharge_delay_min"]   = m("Disposition decision","Departure from ED")
    ct["group"] = np.where(ct.outcome=="LWBS","LWBS",np.where(ct.admitted,"Admitted","Non-admitted"))
    return ct

KPI_COLS = ["door_to_triage_min","door_to_PIA_min","boarding_min","LOS_min"]
SEGMENTS = [("All", lambda c: c.index==c.index)] + \
           [(f"CTAS {k}", (lambda k: lambda c: c.ctas==k)(k)) for k in range(1,6)] + \
           [("Admitted", lambda c: c.group=="Admitted"), ("Non-admitted", lambda c: c.group=="Non-admitted")]

def summarize(ct):
    rows = []
    for k in KPI_COLS:
        for lab, f in SEGMENTS:
            x = ct.loc[f(ct), k].dropna()
            rows.append(dict(kpi=k, segment=lab, n=len(x), mean=x.mean() if len(x) else np.nan,
                             median=x.median() if len(x) else np.nan, p90=x.quantile(.9) if len(x) else np.nan))
    return pd.DataFrame(rows)

def bed_occupancy(ct, n_beds, step_min=15):
    """Occupancy of ED beds (bed assigned -> departure; boarders included), sampled on a grid."""
    b = ct.dropna(subset=["ED bed assigned","departure_ts"])
    t0 = ct.arrival_ts.min().ceil("D") + pd.Timedelta(days=2); t1 = ct.arrival_ts.max() - pd.Timedelta(days=1)
    grid = pd.date_range(t0, t1, freq=f"{step_min}min").values
    st = np.sort(b["ED bed assigned"].values); en = np.sort(b["departure_ts"].values)
    occ = np.searchsorted(st, grid, side="right") - np.searchsorted(en, grid, side="right")
    return dict(mean_occ=occ.mean(), mean_util=occ.mean()/n_beds, pct_time_full=(occ>=n_beds).mean())

def headline(ct):
    seen = ct[ct.outcome!="LWBS"]
    days = max((ct.arrival_ts.max()-ct.arrival_ts.min()).total_seconds()/86400, 1)
    return dict(visits_per_day=len(ct)/days, lwbs_pct=100*(ct.outcome=="LWBS").mean(),
                admit_pct_of_seen=100*seen.admitted.mean() if len(seen) else np.nan)
