"""Calibrate quantities the event log cannot reveal:
  k_reassess      - scales physician reassessment time so the simulated 'work-up done -> disposition decision' interval
                    matches the observed mean (absorbs nurse care + physician reassessment time)
  ward_los_mean   - mean inpatient length of stay so simulated boarding time matches observed (needs ADT data in real life)."""
import numpy as np
from .model import Sim
from .runner import run_once

def _bisect(f, lo, hi, target, iters, log=print, name=""):
    for i in range(iters):
        mid = (lo + hi) / 2; v = f(mid)
        log(f"  {name} iter {i+1}: x={mid:.3f} -> {v:.1f} (target {target:.1f})")
        if v < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def calibrate(P, cfg, days=45, seeds=(11, 12, 13), log=print):
    tgt = P["targets"]; P = {**P, "calibration": {**P["calibration"], "k_reassess": dict(P["calibration"]["k_reassess"])}}
    def sim_metric(key, name, d=None):
        return float(np.mean([run_once((P, cfg, {}, s, d or days, False))[key] for s in seeds]))
    # Single shared scale k (per-CTAS k was tried and over-fits: the observed interval includes endogenous
    # queueing, so matching each CTAS mean pushes physician load unrealistically high).
    log("Calibrating k_reassess (mean decision interval)")
    def f_k(k):
        P["calibration"]["k_reassess"] = {c: float(k) for c in "12345"}
        return sim_metric("decision_interval_mean", "k")
    k = _bisect(f_k, 0.03, 1.0, tgt["mean_decision_interval"], 8, log, "k")
    P["calibration"]["k_reassess"] = {c: float(k) for c in "12345"}
    log("Calibrating ward LOS (boarding time)")
    def f_w(d): P["calibration"]["ward_los_mean_min"] = d * 1440; return sim_metric("boarding_mean", "w", d=days * 2)
    P["calibration"]["ward_los_mean_min"] = _bisect(f_w, 3.0, 5.5, tgt["mean_boarding"], 8, log, "ward_days") * 1440
    return P
