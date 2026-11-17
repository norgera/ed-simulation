"""Fit model parameters from an event log: process map (directly-follows), arrivals, branch probabilities,
service-time distributions (lognormal), LWBS patience. Nothing here is read from the generator."""
import json, numpy as np, pandas as pd
from .kpis import case_table

def _ln(x, floor=0.05):
    x = np.asarray(pd.Series(x).dropna()); x = x[x > 0]
    if len(x) < 5: return None
    l = np.log(np.maximum(x, floor)); return [float(l.mean()), float(l.std(ddof=1))]

def _by_ctas(ct, col, mask=None, min_n=30):
    pooled = _ln(ct.loc[mask, col] if mask is not None else ct[col])
    out = {}
    for k in range(1, 6):
        s = ct[ct.ctas == k]; s = s[mask.reindex(s.index, fill_value=False)] if mask is not None else s
        v = _ln(s[col]) if s[col].notna().sum() >= min_n else None
        out[str(k)] = v or pooled
    return out

def directly_follows(ev):
    e = ev.sort_values(["case_id","timestamp"]).copy()
    e["next"] = e.groupby("case_id").activity.shift(-1)
    d = e.dropna(subset=["next"]).groupby(["activity","next"]).size().reset_index(name="count")
    return d.sort_values("count", ascending=False)

def fit(ev, cs):
    ct = case_table(ev, cs)
    seen = ct[ct.outcome != "LWBS"]
    P = {}
    # arrivals: base rate x hour-of-day profile x day-of-week factor (multiplicative, from marginals)
    a = ct.arrival_ts
    start, end = a.min().normalize(), a.max().normalize() + pd.Timedelta(days=1)
    days = pd.date_range(start, end - pd.Timedelta(days=1))
    n_dow = pd.Series(days.dayofweek).value_counts().reindex(range(7)).values
    c_dow = a.dt.dayofweek.value_counts().reindex(range(7), fill_value=0).values
    c_hr = a.dt.hour.value_counts().reindex(range(24), fill_value=0).values
    total_days = len(days)
    base = len(a) / (total_days * 1440)
    dow_f = (c_dow / n_dow) / (len(a) / total_days)
    hr_f = (c_hr / total_days) / (len(a) / total_days / 24)
    P["arrival"] = dict(base_per_min=base, dow_factor=dow_f.tolist(), hour_factor=hr_f.tolist(),
                        mean_per_day=len(a) / total_days)
    P["ctas_mix"] = {str(k): float((ct.ctas == k).mean()) for k in range(1, 6)}
    # service / delay distributions (lognormal, per CTAS)
    P["triage"]      = _by_ctas(ct.assign(_d=(ct["Triage end"] - ct["Triage start"]).dt.total_seconds()/60), "_d")
    P["pia"]         = _by_ctas(ct.assign(_d=(ct["Physician initial assessment end"] - ct["Physician initial assessment start"]).dt.total_seconds()/60), "_d")
    P["lab_tat"]     = _by_ctas(ct.assign(_d=(ct["Lab result"] - ct["Lab ordered/collected"]).dt.total_seconds()/60), "_d")
    P["di_total"]    = _by_ctas(ct.assign(_d=(ct["DI result"] - ct["DI start"]).dt.total_seconds()/60), "_d")
    P["consult"]     = _by_ctas(ct.assign(_d=(ct["Consult completed"] - ct["Consult requested"]).dt.total_seconds()/60), "_d")
    P["decision_interval"] = _by_ctas(ct, "decision_interval_min", min_n=30)
    P["discharge_delay"]   = _by_ctas(ct[~ct.admitted.astype(bool)], "discharge_delay_min")
    # branch probabilities (of patients seen by a physician)
    def p(col): return {str(k): float(seen[seen.ctas == k][col].notna().mean()) for k in range(1, 6)}
    P["p_lab"], P["p_di"], P["p_consult"] = p("Lab result"), p("DI result"), p("Consult completed")
    P["p_admit"] = {str(k): float(seen[seen.ctas == k].admitted.astype(bool).mean()) for k in range(1, 6)}
    # LWBS patience: censored-exponential MLE = total exposure / #events  (exposure = arrival -> bed or LWBS)
    exp_end = ct["ED bed assigned"].fillna(ct["Left without being seen"])
    expo = (exp_end - ct["Arrival"]).dt.total_seconds() / 60
    pat = {}
    for k in range(1, 6):
        m = ct.ctas == k; n_ev = int((ct[m].outcome == "LWBS").sum())
        pat[str(k)] = float(expo[m].sum() / n_ev) if n_ev >= 5 else None
    P["patience_mean_min"] = pat
    # calibration targets (unobservable quantities get calibrated, see calibrate.py)
    P["targets"] = dict(mean_decision_interval=float(seen.decision_interval_min.mean()),
                        decision_interval_by_ctas={str(k): float(seen[seen.ctas == k].decision_interval_min.mean()) for k in range(1, 6)},
                        mean_boarding=float(ct.boarding_min.mean()))
    P["calibration"] = dict(k_reassess={str(k): 0.3 for k in range(1, 6)}, ward_los_mean_min=4.0 * 1440)
    P["process_map"] = directly_follows(ev).head(40).to_dict("records")
    return P

def save(P, path): json.dump(P, open(path, "w"), indent=1)
def load(path): return json.load(open(path))
