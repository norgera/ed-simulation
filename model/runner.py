"""Run replications and reduce each to a flat metrics dict (picklable, for multiprocessing)."""
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .model import Sim
from .kpis import case_table, headline, bed_occupancy

def metrics(ev, cs, u, cfg_beds):
    ct = case_table(ev, cs); seen = ct[ct.outcome != "LWBS"]; h = headline(ct)
    q = lambda x, p: float(np.nanquantile(x, p)) if len(x.dropna()) else np.nan
    pb = lambda s: s.dropna()
    adm, non = ct[ct.group == "Admitted"], ct[ct.group == "Non-admitted"]
    return dict(**h,
        pia_median=q(ct.door_to_PIA_min, .5), pia_p90=q(ct.door_to_PIA_min, .9), pia_mean=float(ct.door_to_PIA_min.mean()),
        los_median_all=q(ct.LOS_min, .5), los_p90_all=q(ct.LOS_min, .9),
        los_median_non=q(non.LOS_min, .5), los_p90_non=q(non.LOS_min, .9),
        los_median_adm=q(adm.LOS_min, .5), los_p90_adm=q(adm.LOS_min, .9),
        boarding_mean=float(adm.boarding_min.mean()), boarding_p90=q(adm.boarding_min, .9),
        pct_board_gt4h=100 * float((adm.boarding_min > 240).mean()) if len(adm) else np.nan,
        throughput_per_day=float(ct.departure_ts.notna().sum() / max((ct.arrival_ts.max() - ct.arrival_ts.min()).days, 1)),
        util_ed_beds=float(u.bed_util.mean()), pct_time_beds_full=float((u.bed_util >= 0.999).mean()),
        util_physician=float(u.phys_busy.sum() / u.phys_staffed.sum()),
        util_triage=float(u.triage_busy.sum() / u.triage_staffed.sum()),
        util_di=float(u.di_util.mean()), util_ward=float(u.ward_util.mean()),
        mean_q_waiting_bed=float(u.q_bed.mean()), max_q_waiting_bed=float(u.q_bed.max()),
        decision_interval_mean=float(seen.decision_interval_min.mean()),
        **{f'dec_int_c{k}': float(seen[seen.ctas == k].decision_interval_min.mean()) for k in range(1, 6)})

def run_once(args):
    P, cfg, scen, seed, days, full = args
    ev, cs, u = Sim(P, cfg, scen, seed, days).run()
    if full: return ev, cs, u
    return metrics(ev, cs, u, None)

def replicate(P, cfg, scen, seeds, days, workers=2, full=False):
    jobs = [(P, cfg, scen, s, days, full) for s in seeds]
    if workers <= 1: return [run_once(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex: return list(ex.map(run_once, jobs))
