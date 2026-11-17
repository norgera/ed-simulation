"""python -m model all --log data/ed_event_log.csv --cases data/ed_cases.csv --out results"""
import argparse, json, os, time, pandas as pd
from . import kpis, fit, calibrate, validate, scenarios, report
from .model import DEFAULT_CFG

def main(argv=None):
    ap = argparse.ArgumentParser(prog="model")
    ap.add_argument("cmd", choices=["all", "fit", "validate", "scenarios"])
    ap.add_argument("--log", default="data/ed_event_log.csv"); ap.add_argument("--cases", default="data/ed_cases.csv")
    ap.add_argument("--cfg", help="JSON file overriding resource config (staffing, beds)"); ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--quick", action="store_true", help="fewer/shorter replications")
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    cfg = {**DEFAULT_CFG, **(json.load(open(a.cfg)) if a.cfg else {})}
    days, seeds = (40, (101, 102)) if a.quick else (120, (101, 102, 103, 104, 105, 106))
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    log("loading + fitting"); ev, cs = kpis.load(a.log, a.cases); obs = kpis.case_table(ev, cs)
    P = fit.fit(ev, cs); fit.save(P, f"{a.out}/params_fitted.json")
    if a.cmd == "fit": return
    log("calibrating unobserved parameters"); P = calibrate.calibrate(P, cfg, days=30 if a.quick else 45, log=log); fit.save(P, f"{a.out}/params_calibrated.json")
    json.dump(cfg, open(f"{a.out}/resource_config.json", "w"), indent=1)
    log("validating baseline"); val, ks, sims = validate.validate(P, cfg, obs, seeds, days, a.workers)
    val.to_csv(f"{a.out}/validation.csv", index=False); log(f"  {int(val.ok.sum())}/{len(val)} checks pass; KS={ks}")
    if a.cmd == "validate": return
    log("running scenarios"); scen = scenarios.run_scenarios(P, cfg, seeds=seeds, days=days, workers=a.workers, log=log)
    scen.to_csv(f"{a.out}/scenario_results.csv"); scenarios.delta_table(scen).to_csv(f"{a.out}/scenario_deltas.csv")
    report.build(f"{a.out}/report.html", P, cfg, val, ks, obs, sims, scen); log(f"report -> {a.out}/report.html")

if __name__ == "__main__": main()
