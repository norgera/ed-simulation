"""python -m engine run --log events.csv --out results      (add --spec spec.json to override the drafted resource spec)"""
import argparse, json, os, time
import pandas as pd
from .log import load_log
from .discover import discover, mermaid, merged_map, mermaid_merged
from .spec import draft_spec, load_spec, save_spec
from .fit import fit
from .validate import validate

def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine")
    ap.add_argument("cmd", choices=["run", "draft-spec", "whatif"])
    ap.add_argument("--log"); ap.add_argument("--out", default="results")
    ap.add_argument("--case-col", default="case_id"); ap.add_argument("--activity-col", default="activity")
    ap.add_argument("--time-col", default="timestamp"); ap.add_argument("--resource-col", default="resource")
    ap.add_argument("--cases", help="optional table with one row per case (for the class column)")
    ap.add_argument("--class-col", help="column that classifies cases (e.g. acuity level); optional")
    ap.add_argument("--spec", help="resource spec JSON; drafted from the log when omitted")
    ap.add_argument("--calibrate", help="pool name whose hidden-work share to calibrate (needs a `visits` entry in the spec)")
    ap.add_argument("--params", help="fitted params.json (for whatif)"); ap.add_argument("--scenarios", help="scenarios JSON (for whatif)")
    ap.add_argument("--name", help="title for the report (default: the case folder name)")
    ap.add_argument("--loop", action="store_true", help="validation loop: re-tune unknowns until checks are within --tol or stop improving")
    ap.add_argument("--warmup", type=int, default=10, help="simulated days discarded before measuring (use ~30 when a ward or other long stay is modelled)")
    ap.add_argument("--tol", type=float, default=0.05); ap.add_argument("--reps", type=int, default=4); ap.add_argument("--days", type=int, default=60)
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    say = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    if a.cmd == "whatif":
        from .scenarios import run_scenarios, load_scenarios
        P = json.load(open(a.params)); spec = load_spec(a.spec)
        df = run_scenarios(P, spec, load_scenarios(a.scenarios), reps=a.reps, days=a.days, log=print)
        df.to_csv(f"{a.out}/scenario_results.csv"); print(f"scenario results -> {a.out}/scenario_results.csv"); return
    log, classes = load_log(a.log, a.case_col, a.activity_col, a.time_col, a.resource_col, a.cases, a.class_col)
    say(f"loaded {len(log)} events, {log.case_id.nunique()} cases, {log.activity.nunique()} activities")
    pm = discover(log); json.dump(pm, open(f"{a.out}/process_discovery.json", "w"), indent=1)
    open(f"{a.out}/process_map_raw.mmd", "w").write(mermaid(pm))
    open(f"{a.out}/process_map.mmd", "w").write(mermaid_merged(merged_map(log))); say("process map -> process_map.mmd (start/end merged), process_map_raw.mmd")
    spec = load_spec(a.spec) if a.spec else draft_spec(log)
    if not a.spec:
        save_spec(spec, f"{a.out}/spec_drafted.json"); say(f"drafted spec: {len(spec['pools'])} pool(s) -> spec_drafted.json")
        for n in spec["notes"]: say("  note: " + n)
    if a.cmd == "draft-spec": return
    P = fit(log, classes, spec); json.dump(P, open(f"{a.out}/params.json", "w"))
    if a.calibrate:
        from .calibrate import calibrate_visit_scale
        from .kpis import summarize
        say(f"calibrating hidden work for '{a.calibrate}'")
        P, k = calibrate_visit_scale(P, spec, summarize(log, classes, spec, set(P["terminals"])), a.calibrate, log=lambda m: say(m))
        say(f"  -> hidden-work share = {k:.3f}"); json.dump(P, open(f"{a.out}/params.json", "w"))
    say("fitted; validating")
    if a.loop:
        from .calibrate import validation_loop
        P, val, summ, hist, sim_logs = validation_loop(P, spec, log, classes, a.tol, reps=a.reps, days=a.days, warmup_days=a.warmup, log=say)
        summ["loop_history"] = hist; json.dump(P, open(f"{a.out}/params.json", "w"))
    else:
        val, summ, sim_logs = validate(P, spec, log, classes, a.reps, a.days, a.warmup, tol=a.tol, keep_logs=True)
    val.to_csv(f"{a.out}/validation.csv", index=False); json.dump(summ, open(f"{a.out}/validation_summary.json", "w"), indent=1)
    c = summ["checks"]
    say(f"{summ['within_5pct']}/{c} within 5%, {summ['within_10pct']}/{c} within 10%, {summ['within_20pct']}/{c} within 20%")
    print(val[val.status.isin(["FAIL", "NOISE"])].round(3).to_string(index=False))
    from .report import build
    case_dir = os.path.dirname(os.path.abspath(a.out)); cj = os.path.join(case_dir, "case.json")
    title = a.name or (json.load(open(cj)).get("title") if os.path.exists(cj) else None) or os.path.basename(case_dir)
    build(f"{a.out}/report.html", title, log, classes, spec, P, val, summ, sim_logs)
    say(f"report -> {a.out}/report.html")

if __name__ == "__main__": main()
