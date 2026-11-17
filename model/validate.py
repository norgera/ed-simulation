"""Validate the baseline model against the observed log: same KPI code on both, replications, tolerance checks."""
import numpy as np, pandas as pd
from scipy import stats
from .runner import replicate
from .kpis import case_table, summarize, bed_occupancy, headline

# relative-error tolerances for acceptance
TOL = dict(mean=0.20, median=0.15, p90=0.25)
CHECK_KPIS = [("door_to_PIA_min", "All"), ("door_to_PIA_min", "CTAS 2"), ("door_to_PIA_min", "CTAS 3"), ("door_to_PIA_min", "CTAS 4"),
              ("LOS_min", "All"), ("LOS_min", "Admitted"), ("LOS_min", "Non-admitted"), ("boarding_min", "Admitted")]

def validate(P, cfg, obs_ct, seeds=(101, 102, 103, 104), days=120, workers=2):
    runs = replicate(P, cfg, {}, seeds, days, workers, full=True)
    sims = [case_table(ev, cs) for ev, cs, u in runs]
    sim_tabs = [summarize(c) for c in sims]
    obs_tab = summarize(obs_ct)
    rows = []
    for kpi, seg in CHECK_KPIS:
        o = obs_tab[(obs_tab.kpi == kpi) & (obs_tab.segment == seg)].iloc[0]
        S = pd.concat([t[(t.kpi == kpi) & (t.segment == seg)] for t in sim_tabs])
        for stat in (["mean", "p90"] if kpi == "boarding_min" else ["mean", "median", "p90"]):   # boarding median is 0 (bimodal) -> not a useful check
            sv, ov = S[stat].mean(), o[stat]
            if kpi == "boarding_min" and stat == "median" : tol = 0.60   # boarding is bimodal (0 or long): median is unstable
            else: tol = TOL[stat]
            if kpi == "door_to_PIA_min" and stat != "p90": tol = max(tol, 0.30)   # tiny values: relative error inflated
            err = (sv - ov) / ov if ov else np.nan
            rows.append(dict(kpi=kpi, segment=seg, stat=stat, observed=ov, simulated=sv, sim_sd=S[stat].std(),
                             rel_err=err, tol=tol, ok=bool(abs(err) <= tol)))
    # headline + utilisation
    hl_obs = headline(obs_ct); hl_sim = pd.DataFrame([headline(c) for c in sims]).mean()
    occ_obs = bed_occupancy(obs_ct, cfg.get("ed_beds", 30)); occ_sim = pd.DataFrame([bed_occupancy(c, cfg.get("ed_beds", 30)) for c in sims]).mean()
    extra = [("visits_per_day", hl_obs["visits_per_day"], hl_sim["visits_per_day"], .05),
             ("lwbs_pct", hl_obs["lwbs_pct"], hl_sim["lwbs_pct"], None), ("admit_pct_of_seen", hl_obs["admit_pct_of_seen"], hl_sim["admit_pct_of_seen"], .10),
             ("ed_bed_mean_util", occ_obs["mean_util"], occ_sim["mean_util"], .10),
             ("ed_bed_pct_time_full", occ_obs["pct_time_full"], occ_sim["pct_time_full"], None)]
    for name, ov, sv, tol in extra:
        if tol is None:   # absolute tolerance: 2 points / 5 points
            err = sv - ov; lim = 2.0 if name == "lwbs_pct" else 0.08; ok = abs(err) <= lim; tol = lim; relerr = err
        else: relerr = (sv - ov) / ov; ok = abs(relerr) <= tol
        rows.append(dict(kpi=name, segment="All", stat="value", observed=ov, simulated=sv, sim_sd=np.nan, rel_err=relerr, tol=tol, ok=bool(ok)))
    res = pd.DataFrame(rows)
    # distribution comparison (KS) on pooled simulated vs observed
    ks = {}
    for col in ["door_to_PIA_min", "LOS_min"]:
        a = obs_ct[col].dropna(); b = pd.concat([c[col].dropna() for c in sims])
        ks[col] = float(stats.ks_2samp(a, b).statistic)
    return res, ks, sims
