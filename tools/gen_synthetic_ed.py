"""
Synthetic ED event-log generator (SimPy discrete-event model).

All parameters below are ASSUMED, chosen to be plausible for a busy Canadian
urban ED. They are NOT fitted to any real hospital. Replace with fitted values
once real data is available. Output is long-format event log + case table.

Usage: python gen_synthetic_ed.py --days 365 --seed 1 --out outdir
       python gen_synthetic_ed.py --zones --out cases/ed_zones/data      # zone-based ED (see EDZones)
"""
import argparse, os
import numpy as np, pandas as pd, simpy

CTAS_MIX   = {1: .01, 2: .13, 3: .42, 4: .34, 5: .10}
ADMIT_P    = {1: .80, 2: .40, 3: .16, 4: .03, 5: .01}
LAB_P      = {1: .95, 2: .85, 3: .60, 4: .15, 5: .03}
DI_P       = {1: .80, 2: .60, 3: .40, 4: .20, 5: .05}
CONSULT_P  = {1: .60, 2: .35, 3: .12, 4: .02, 5: .0}
PIA_MEAN   = {1: 25, 2: 24, 3: 21, 4: 16, 5: 12}     # min, physician time
REASS_MEAN = {1: 20, 2: 14, 3: 11, 4: 7,  5: 5}      # min, physician time
PATIENCE   = {1: 1e9, 2: 1e9, 3: 600, 4: 300, 5: 240}  # mean min before LWBS
HOURLY = np.array([.55,.45,.38,.33,.32,.35,.45,.65,.90,1.10,1.20,1.25,
                   1.25,1.20,1.20,1.20,1.20,1.25,1.25,1.15,1.05,.90,.75,.62])
DOW    = np.array([1.08,1.0,.97,.95,.97,.98,1.05])    # Mon..Sun
BASE_RATE = 7.2 / 60.0                                 # mean patients / min
# resources: (day 07-15, evening 15-23, night 23-07)
CAP = dict(triage=(3,3,2), physician=(4,4,2), nurse=(9,9,7))
N_BEDS, N_DI, N_LAB, N_WARD = 30, 2, 3, 93
WARD_LOS_MEAN_D = 4.2

def lognorm(rng, mean, cv):
    s2 = np.log(1+cv**2); return rng.lognormal(np.log(mean)-s2/2, np.sqrt(s2))

class ED:
    def __init__(s, env, rng, start):
        s.env, s.rng, s.start = env, rng, start
        s.triage = simpy.PriorityResource(env, 3)
        s.phys   = simpy.PriorityResource(env, 4)
        s.nurse  = simpy.PriorityResource(env, 9)
        s.beds   = simpy.PriorityResource(env, N_BEDS)
        s.di     = simpy.PriorityResource(env, N_DI)
        s.lab    = simpy.PriorityResource(env, N_LAB)
        s.ward   = simpy.PriorityResource(env, N_WARD)
        s.events, s.cases, s.active_beds = [], [], 0
        env.process(s.shifts())
        env.process(s.ward_background())
    def ts(s, t): return s.start + pd.Timedelta(minutes=float(t))
    def log(s, cid, act, res=""): s.events.append((cid, act, s.ts(s.env.now), res))
    def shifts(s):
        # shift-dependent capacity via high-priority blocker requests
        res = dict(triage=s.triage, physician=s.phys, nurse=s.nurse)
        full = dict(triage=3, physician=4, nurse=9)
        blockers = {k: [] for k in res}
        while True:
            h = (s.env.now/60) % 24
            idx = 0 if 7<=h<15 else 1 if 15<=h<23 else 2
            for k, r in res.items():
                want = full[k]-CAP[k][idx]
                while len(blockers[k]) > want:
                    r.release(blockers[k].pop())
                while len(blockers[k]) < want:
                    rq = r.request(priority=-10); blockers[k].append(rq)
            yield s.env.timeout(60)
    def ward_background(s):
        # pre-existing inpatients occupying ~80% of ward beds at t=0
        for _ in range(int(N_WARD*0.85)):
            s.env.process(s.ward_stay(None, initial=True))
        yield s.env.timeout(0)
    def ward_stay(s, cid, initial=False):
        with s.ward.request(priority=1) as rq:
            yield rq
            if cid: s.log(cid, "Inpatient bed assigned", "ward")
            los = s.rng.exponential(WARD_LOS_MEAN_D*1440) if initial \
                  else lognorm(s.rng, WARD_LOS_MEAN_D*1440, 0.9)
            yield s.env.timeout(los)
        return
    def ward_hold(s, wrq):
        yield s.env.timeout(lognorm(s.rng, WARD_LOS_MEAN_D*1440, 0.9))
        s.ward.release(wrq)
    def patient(s, cid):
        r, e = s.rng, s.env
        ctas = int(r.choice(list(CTAS_MIX), p=list(CTAS_MIX.values())))
        pr = ctas
        arrive = e.now
        mode = r.choice(["Ambulance","Walk-in"], p=[.9,.1] if ctas==1 else [.35,.65] if ctas==2 else [.12,.88])
        s.log(cid, "Arrival", mode)
        # triage + registration
        with s.triage.request(priority=min(pr,2)) as q:
            yield q
            s.log(cid, "Triage start", "triage nurse")
            yield e.timeout(lognorm(r, 5.0, .5))
            s.log(cid, "Triage end", "triage nurse")
        # wait for ED bed AND physician; may leave without being seen
        pat = r.exponential(PATIENCE[ctas])
        bed_rq = s.beds.request(priority=pr)
        res = yield bed_rq | e.timeout(max(pat - (e.now-arrive), 0.01))
        if bed_rq not in res:
            bed_rq.cancel(); s.log(cid, "Left without being seen")
            s.cases.append(dict(case_id=cid, ctas=ctas, arrival_mode=mode, outcome="LWBS",
                                admitted=False, arrive=arrive, depart=e.now)); return
        s.log(cid, "ED bed assigned", "ED bed")
        with s.phys.request(priority=pr) as q:
            yield q
            s.log(cid, "Physician initial assessment start", "physician")
            yield e.timeout(lognorm(r, PIA_MEAN[ctas], .6))
            s.log(cid, "Physician initial assessment end", "physician")
        pia_t = e.now
        # diagnostics (nurse draws / transport; lab + DI processing; turnaround delays)
        need_lab = r.random() < LAB_P[ctas]; need_di = r.random() < DI_P[ctas]
        def lab():
            with s.nurse.request(priority=pr) as q:
                yield q; yield e.timeout(lognorm(r, 6, .4))
            s.log(cid, "Lab ordered/collected", "nurse")
            with s.lab.request(priority=pr) as q:
                yield q; yield e.timeout(lognorm(r, 8, .4))
            yield e.timeout(lognorm(r, 55 if ctas<=2 else 80, .6))
            s.log(cid, "Lab result", "lab")
        def di():
            with s.di.request(priority=pr) as q:
                yield q
                s.log(cid, "DI start", "DI scanner")
                yield e.timeout(lognorm(r, 20, .5))
            yield e.timeout(lognorm(r, 45, .8))
            s.log(cid, "DI result", "radiology")
        procs = ([e.process(lab())] if need_lab else []) + ([e.process(di())] if need_di else [])
        if procs: yield simpy.AllOf(e, procs)
        if r.random() < CONSULT_P[ctas]:
            s.log(cid, "Consult requested")
            yield e.timeout(lognorm(r, 110, .8))
            s.log(cid, "Consult completed", "consultant")
        # nursing care (meds, monitoring) concurrent with waiting - modelled as one block
        with s.nurse.request(priority=pr) as q:
            yield q; yield e.timeout(lognorm(r, 15 if ctas<=3 else 8, .5))
        # reassessment + disposition decision
        with s.phys.request(priority=pr) as q:
            yield q; yield e.timeout(lognorm(r, REASS_MEAN[ctas], .6))
        adm = r.random() < ADMIT_P[ctas]
        s.log(cid, "Disposition decision", "physician")
        s.cases.append(dict(case_id=cid, ctas=ctas, arrival_mode=mode, outcome="ADMITTED" if adm else "DISCHARGED",
                            admitted=adm, arrive=arrive, depart=None))
        c = s.cases[-1]
        if adm:  # boarding: holds ED bed until an inpatient bed is free
            s.log(cid, "Boarding start (admit request)")
            wrq = s.ward.request(priority=1); yield wrq
            s.log(cid, "Inpatient bed assigned", "ward")
            e.process(s.ward_hold(wrq))
        else:
            yield e.timeout(lognorm(r, 25, .7))   # discharge paperwork / teaching
        s.log(cid, "Departure from ED")
        c["depart"] = e.now
        s.beds.release(bed_rq)

# ---- zone-based ED (placeholder layout until the supervisor's ideal ED process map arrives; every number is ASSUMED)
# After triage each patient goes to a zone. Only patients who cannot stay vertical get a stretcher; vertical patients
# are seen in chairs and move to a stretcher only if they are admitted (they board on a stretcher).
ZONES = {"red":    dict(space="stretcher", n=8,  note="resuscitation: CTAS 1, CTAS 2 by ambulance"),
         "yellow": dict(space="stretcher", n=22, note="acute, cannot stay vertical: other ambulance arrivals, CTAS 2 walk-ins, 10% of CTAS 3 walk-ins"),
         "blue":   dict(space="chair",     n=14, note="vertical, results pending: CTAS 3 walk-ins"),
         "green":  dict(space="chair",     n=10, note="fast track: CTAS 4-5 walk-ins, own provider")}
# staffing by start hour: main physicians cover red/yellow/blue, a fast-track provider covers green
ZCAP = dict(triage=[(7, 3), (15, 3), (23, 2)], physician=[(7, 3), (15, 3), (23, 2)],
            fasttrack=[(7, 1), (11, 2), (21, 1)], nurse=[(7, 9), (15, 9), (23, 7)])

def zone_for(ctas, mode, r):
    if ctas == 1 or (ctas == 2 and mode == "Ambulance"): return "red"
    if mode == "Ambulance" or ctas == 2: return "yellow"
    if ctas == 3: return "yellow" if r.random() < .10 else "blue"
    return "green"

def cap_at(sched, h):
    c = sorted(sched)[-1][1]
    for start, n in sorted(sched):
        if h >= start: c = n
    return c

class EDZones(ED):
    def __init__(s, env, rng, start):
        s.env, s.rng, s.start = env, rng, start
        mx = {k: max(n for _, n in v) for k, v in ZCAP.items()}
        s.triage = simpy.PriorityResource(env, mx["triage"]); s.phys = simpy.PriorityResource(env, mx["physician"])
        s.ft = simpy.PriorityResource(env, mx["fasttrack"]); s.nurse = simpy.PriorityResource(env, mx["nurse"])
        s.space = {z: simpy.PriorityResource(env, c["n"]) for z, c in ZONES.items()}
        s.di = simpy.PriorityResource(env, N_DI); s.lab = simpy.PriorityResource(env, N_LAB); s.ward = simpy.PriorityResource(env, N_WARD)
        s.events, s.cases = [], []
        env.process(s.shifts()); env.process(s.ward_background())
    def shifts(s):
        res = dict(triage=s.triage, physician=s.phys, fasttrack=s.ft, nurse=s.nurse); blockers = {k: [] for k in res}
        while True:
            h = (s.env.now / 60) % 24
            for k, r in res.items():
                want = r.capacity - cap_at(ZCAP[k], h)
                while len(blockers[k]) > want: r.release(blockers[k].pop())
                while len(blockers[k]) < want: blockers[k].append(r.request(priority=-10))
            yield s.env.timeout(60)
    def patient(s, cid):
        r, e = s.rng, s.env
        ctas = int(r.choice(list(CTAS_MIX), p=list(CTAS_MIX.values()))); pr = ctas; arrive = e.now
        mode = r.choice(["Ambulance","Walk-in"], p=[.9,.1] if ctas==1 else [.35,.65] if ctas==2 else [.12,.88])
        s.log(cid, "Arrival", mode)
        with s.triage.request(priority=min(pr, 2)) as q:
            yield q; s.log(cid, "Triage start", "triage nurse"); yield e.timeout(lognorm(r, 5.0, .5)); s.log(cid, "Triage end", "triage nurse")
        zone = zone_for(ctas, mode, r); kind = ZONES[zone]["space"]; ft = zone == "green"
        case = dict(case_id=cid, ctas=ctas, arrival_mode=mode, zone=zone, ctas_zone=f"{ctas}-{zone}", arrive=arrive, depart=None)
        pat = r.exponential(PATIENCE[ctas]); left = lambda: max(pat - (e.now - arrive), 0.01)
        def lwbs(*held):
            for res_, rq in held: res_.release(rq)
            s.log(cid, "Left without being seen"); s.cases.append({**case, "outcome": "LWBS", "admitted": False, "depart": e.now})
        sp = s.space[zone]; sp_rq = sp.request(priority=pr)        # patience runs from arrival through the wait for a physician
        got = yield sp_rq | e.timeout(left())
        if sp_rq not in got: sp_rq.cancel(); lwbs(); return
        s.log(cid, f"{zone.capitalize()} {kind} assigned", f"{zone} {kind}")
        doc = s.ft if ft else s.phys; who = "fast-track provider" if ft else "physician"
        step = "Fast-track assessment" if ft else "Physician initial assessment"
        d_rq = doc.request(priority=pr); got = yield d_rq | e.timeout(left())
        if d_rq not in got: d_rq.cancel(); lwbs((sp, sp_rq)); return
        s.log(cid, f"{step} start", who); yield e.timeout(lognorm(r, PIA_MEAN[ctas], .6)); s.log(cid, f"{step} end", who)
        doc.release(d_rq)
        need_lab = r.random() < LAB_P[ctas]; need_di = r.random() < DI_P[ctas]
        def lab():
            with s.nurse.request(priority=pr) as q:
                yield q; yield e.timeout(lognorm(r, 6, .4))
            s.log(cid, "Lab ordered/collected", "nurse")
            with s.lab.request(priority=pr) as q:
                yield q; yield e.timeout(lognorm(r, 8, .4))
            yield e.timeout(lognorm(r, 55 if ctas <= 2 else 80, .6)); s.log(cid, "Lab result", "lab")
        def di():
            with s.di.request(priority=pr) as q:
                yield q; s.log(cid, "DI start", "DI scanner"); yield e.timeout(lognorm(r, 20, .5))
            yield e.timeout(lognorm(r, 45, .8)); s.log(cid, "DI result", "radiology")
        procs = ([e.process(lab())] if need_lab else []) + ([e.process(di())] if need_di else [])
        if procs: yield simpy.AllOf(e, procs)
        if r.random() < CONSULT_P[ctas]:
            s.log(cid, "Consult requested"); yield e.timeout(lognorm(r, 110, .8)); s.log(cid, "Consult completed", "consultant")
        with s.nurse.request(priority=pr) as q:
            yield q; yield e.timeout(lognorm(r, 15 if ctas <= 3 else 8, .5))
        with doc.request(priority=pr) as q:                         # reassessment by the same kind of provider
            yield q; yield e.timeout(lognorm(r, REASS_MEAN[ctas], .6))
        adm = r.random() < ADMIT_P[ctas]; s.log(cid, "Disposition decision", who)
        case.update(outcome="ADMITTED" if adm else "DISCHARGED", admitted=adm); s.cases.append(case)
        if adm:
            if kind == "chair":                                       # admitted vertical patients board on a stretcher
                st = s.space["yellow"]; st_rq = st.request(priority=pr); yield st_rq
                s.log(cid, "Yellow stretcher assigned", "yellow stretcher"); sp.release(sp_rq); sp, sp_rq = st, st_rq
            s.log(cid, "Boarding start (admit request)")
            wrq = s.ward.request(priority=1); yield wrq
            s.log(cid, "Inpatient bed assigned", "ward"); e.process(s.ward_hold(wrq))
        else:
            yield e.timeout(lognorm(r, 25, .7))
        s.log(cid, "Departure from ED"); case["depart"] = e.now; sp.release(sp_rq)

def run(days, seed, warmup_days=14, zones=False):
    rng = np.random.default_rng(seed)
    env = simpy.Environment(); start = pd.Timestamp("2025-01-01") - pd.Timedelta(days=warmup_days)
    ed = (EDZones if zones else ED)(env, rng, start)
    total = (days+warmup_days)*1440
    def arrivals():
        cid = 0
        lam_max = BASE_RATE*HOURLY.max()*DOW.max()
        while env.now < total:
            yield env.timeout(rng.exponential(1/lam_max))
            t = ed.ts(env.now); lam = BASE_RATE*HOURLY[t.hour]*DOW[t.dayofweek]
            if rng.random() < lam/lam_max:
                cid += 1; env.process(ed.patient(f"P{cid:06d}"))
    env.process(arrivals()); env.run(until=total)
    return ed, start, warmup_days

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default=".")
    ap.add_argument("--zones", action="store_true", help="zone-based ED: red/yellow/blue/green zones, stretchers only when needed")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ed, start, wu = run(a.days, a.seed, zones=a.zones)
    ev = pd.DataFrame(ed.events, columns=["case_id","activity","timestamp","resource"])
    cs = pd.DataFrame(ed.cases); cs["arrival_ts"] = cs.arrive.map(ed.ts)
    cs["departure_ts"] = cs.depart.map(lambda x: ed.ts(x) if pd.notna(x) else pd.NaT)
    cut = pd.Timestamp("2025-01-01")
    keep = set(cs.loc[(cs.arrival_ts >= cut) & cs.departure_ts.notna(), "case_id"])  # drop cases still in ED at end (censored)
    cs = cs[cs.case_id.isin(keep)].drop(columns=["arrive","depart"]).sort_values("arrival_ts")
    ev = ev[ev.case_id.isin(keep)].sort_values(["timestamp","case_id"])
    cs.to_csv(os.path.join(a.out,"ed_cases.csv"), index=False)
    ev.to_csv(os.path.join(a.out,"ed_event_log.csv"), index=False)
    print(len(cs), "cases;", len(ev), "events")
