"""SimPy discrete-event model of ED patient flow, driven by fitted params + resource config + scenario levers."""
import numpy as np, pandas as pd, simpy

# Resource configuration: NOT inferable from an event log -> must come from the hospital (rosters, bed counts).
DEFAULT_CFG = dict(triage=[3, 3, 2], physician=[4, 4, 2], ed_beds=30, di=2, ward_beds=93,
                   di_scan_min=20.0, ward_init_occ=0.85, ward_los_cv=0.9)

def _shift(h): return 0 if 7 <= h < 15 else 1 if 15 <= h < 23 else 2

class Sim:
    def __init__(s, P, cfg=None, scen=None, seed=1, days=90, warmup_days=14):
        s.P = P; s.cfg = {**DEFAULT_CFG, **(cfg or {})}; s.sc = scen or {}
        s.cfg.update(s.sc.get("cfg", {}))
        s.rng = np.random.default_rng(seed); s.env = simpy.Environment()
        s.days, s.wu = days, warmup_days
        s.start = pd.Timestamp("2025-01-01") - pd.Timedelta(days=warmup_days)
        e, c = s.env, s.cfg
        s.triage = simpy.PriorityResource(e, max(c["triage"])); s.phys = simpy.PriorityResource(e, max(c["physician"]))
        s.beds = simpy.PriorityResource(e, c["ed_beds"]); s.di = simpy.PriorityResource(e, c["di"])
        s.ward = simpy.PriorityResource(e, c["ward_beds"])
        s.ev, s.cases, s.util = [], [], []
        s.waiting_bed = 0; s.nblock = {"triage": 0, "phys": 0}
        s.mult = lambda k: float(s.sc.get(k, 1.0))
        cal = P["calibration"]; s.k = {int(c): v for c, v in cal["k_reassess"].items()}
        s.ward_mean = cal["ward_los_mean_min"] * s.mult("ward_los_mult")
        cv = c["ward_los_cv"]; s2 = np.log(1 + cv**2)
        s.ward_mu, s.ward_sig = np.log(s.ward_mean) - s2 / 2, np.sqrt(s2)
        s.mix = np.array([P["ctas_mix"][str(k)] for k in range(1, 6)]); s.mix = s.mix / s.mix.sum()
        if "ctas_mix" in s.sc:
            s.mix = np.array([s.sc["ctas_mix"][str(k)] for k in range(1, 6)]); s.mix = s.mix / s.mix.sum()
        e.process(s.shifts()); e.process(s.monitor()); e.process(s.ward_background()); e.process(s.arrivals())

    # ---- helpers
    def ln(s, key, ctas, mult=1.0):
        mu, sig = s.P[key][str(ctas)]; return s.rng.lognormal(mu, sig) * mult
    def log(s, cid, act, res=""): s.ev.append((cid, act, s.env.now, res))
    def rate(s, t):
        a = s.P["arrival"]; ts = s.start + pd.Timedelta(minutes=t)
        return a["base_per_min"] * a["hour_factor"][ts.hour] * a["dow_factor"][ts.dayofweek] * s.mult("arrival_mult")

    # ---- processes
    def shifts(s):
        held = {"triage": [], "phys": []}; res = {"triage": s.triage, "phys": s.phys}
        cap = {"triage": s.cfg["triage"], "phys": s.cfg["physician"]}
        while True:
            idx = _shift(int((s.env.now / 60) % 24))
            for k, r in res.items():
                want = max(cap[k]) - cap[k][idx]
                while len(held[k]) > want: r.release(held[k].pop())
                while len(held[k]) < want: held[k].append(r.request(priority=-10))
                s.nblock[k] = len(held[k])
            yield s.env.timeout(60)
    def monitor(s):
        while True:
            idx = _shift(int((s.env.now / 60) % 24)); c = s.cfg
            s.util.append((s.env.now, s.beds.count / c["ed_beds"], s.phys.count - s.nblock["phys"], c["physician"][idx],
                           s.triage.count - s.nblock["triage"], c["triage"][idx], s.di.count / c["di"],
                           s.ward.count / c["ward_beds"], s.waiting_bed))
            yield s.env.timeout(5)
    def ward_background(s):
        for _ in range(int(s.cfg["ward_beds"] * s.cfg["ward_init_occ"])):
            s.env.process(s.ward_initial())
        yield s.env.timeout(0)
    def ward_initial(s):
        with s.ward.request(priority=1) as rq:
            yield rq; yield s.env.timeout(s.rng.exponential(s.ward_mean))
    def ward_hold(s, rq):
        yield s.env.timeout(s.rng.lognormal(s.ward_mu, s.ward_sig)); s.ward.release(rq)
    def arrivals(s):
        total = (s.days + s.wu) * 1440; a = s.P["arrival"]
        lam_max = a["base_per_min"] * max(a["hour_factor"]) * max(a["dow_factor"]) * s.mult("arrival_mult")
        n = 0
        while s.env.now < total:
            yield s.env.timeout(s.rng.exponential(1 / lam_max))
            if s.rng.random() < s.rate(s.env.now) / lam_max:
                n += 1; s.env.process(s.patient(f"S{n:06d}"))

    def patient(s, cid):
        P, e, r = s.P, s.env, s.rng
        ctas = int(r.choice([1, 2, 3, 4, 5], p=s.mix)); t0 = e.now
        s.log(cid, "Arrival")
        with s.triage.request(priority=min(ctas, 2)) as q:
            yield q; s.log(cid, "Triage start", "triage nurse")
            yield e.timeout(s.ln("triage", ctas)); s.log(cid, "Triage end", "triage nurse")
        pm = P["patience_mean_min"][str(ctas)]
        bed = s.beds.request(priority=ctas); s.waiting_bed += 1
        if pm:
            yield bed | e.timeout(max(r.exponential(pm) - (e.now - t0), 0.01))
        else:
            yield bed
        s.waiting_bed -= 1
        if not bed.triggered:
            bed.cancel(); s.log(cid, "Left without being seen")
            s.cases.append((cid, ctas, "LWBS", False, t0, e.now)); return
        s.log(cid, "ED bed assigned", "ED bed")
        with s.phys.request(priority=ctas) as q:
            yield q; s.log(cid, "Physician initial assessment start", "physician")
            yield e.timeout(s.ln("pia", ctas, s.mult("pia_mult"))); s.log(cid, "Physician initial assessment end", "physician")
        procs = []
        if r.random() < P["p_lab"][str(ctas)]:
            procs.append(e.process(s.lab(cid, ctas)))
        if r.random() < P["p_di"][str(ctas)]:
            procs.append(e.process(s.imaging(cid, ctas)))
        if procs: yield simpy.AllOf(e, procs)
        if r.random() < P["p_consult"][str(ctas)]:
            s.log(cid, "Consult requested"); yield e.timeout(s.ln("consult", ctas, s.mult("consult_mult")))
            s.log(cid, "Consult completed", "consultant")
        with s.phys.request(priority=ctas) as q:   # reassessment / disposition (unobserved service time -> calibrated k)
            yield q; yield e.timeout(s.ln("decision_interval", ctas, s.k[ctas] * s.mult("reassess_mult")))
        s.log(cid, "Disposition decision", "physician")
        adm = r.random() < min(1.0, P["p_admit"][str(ctas)] * s.mult("admit_mult"))
        if adm:
            s.log(cid, "Boarding start (admit request)")
            w = s.ward.request(priority=1); yield w
            s.log(cid, "Inpatient bed assigned", "ward"); e.process(s.ward_hold(w))
        else:
            yield e.timeout(s.ln("discharge_delay", ctas, s.mult("discharge_mult")))
        s.log(cid, "Departure from ED"); s.beds.release(bed)
        s.cases.append((cid, ctas, "ADMITTED" if adm else "DISCHARGED", adm, t0, e.now))
    def lab(s, cid, ctas):
        s.log(cid, "Lab ordered/collected", "nurse")
        yield s.env.timeout(s.ln("lab_tat", ctas, s.mult("lab_mult"))); s.log(cid, "Lab result", "lab")
    def imaging(s, cid, ctas):
        scan = s.cfg["di_scan_min"] * s.mult("di_mult")
        with s.di.request(priority=ctas) as q:
            yield q; s.log(cid, "DI start", "DI scanner")
            tot = s.ln("di_total", ctas, s.mult("di_mult")); hold = min(scan, tot); yield s.env.timeout(hold)
        yield s.env.timeout(max(tot - hold, 0)); s.log(cid, "DI result", "radiology")

    # ---- output
    def run(s):
        s.env.run(until=(s.days + s.wu) * 1440)
        T = lambda m: s.start + pd.to_timedelta(np.asarray(m, float), unit="m")
        ev = pd.DataFrame(s.ev, columns=["case_id", "activity", "t", "resource"]); ev["timestamp"] = T(ev.t)
        cs = pd.DataFrame(s.cases, columns=["case_id", "ctas", "outcome", "admitted", "a", "d"])
        cs["arrival_ts"], cs["departure_ts"] = T(cs.a), T(cs.d)
        cut = pd.Timestamp("2025-01-01"); keep = set(cs.loc[cs.arrival_ts >= cut, "case_id"])
        cs = cs[cs.case_id.isin(keep)].drop(columns=["a", "d"]); ev = ev[ev.case_id.isin(keep)][["case_id", "activity", "timestamp", "resource"]]
        u = pd.DataFrame(s.util, columns=["t", "bed_util", "phys_busy", "phys_staffed", "triage_busy", "triage_staffed",
                                          "di_util", "ward_util", "q_bed"]); u = u[u.t >= s.wu * 1440]
        return ev, cs, u
