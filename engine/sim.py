"""Discrete-event simulation of a fitted process (SimPy). Output has the same schema as an input event log."""
import heapq, math, re
import numpy as np, pandas as pd, simpy
from .fit import QGRID

class Pool:
    """A resource with time-varying capacity, optional priority by class, and cancellable requests."""
    def __init__(s, env, p):
        s.env, s.p, s.q, s.seq, s.in_use = env, p, [], 0, 0
        s.shifts = sorted(p["shifts"]) if p.get("shifts") else None
        s.area_busy = s.area_cap = 0.0; s.last = 0.0; s.log_from = 0.0
        env.process(s._shift_clock())
    def cap(s):
        if not s.shifts:
            return max(0, int(s.p["capacity"]))
        h = (s.env.now / 60.0 + s.p.get("hour0", 0)) % 24
        c = s.shifts[-1][1]
        for start, cap in s.shifts:
            if h >= start: c = cap
        return max(0, int(c))
    def _tick(s):
        now = s.env.now
        if now > s.log_from:
            dt = now - max(s.last, s.log_from)
            s.area_busy += s.in_use * dt; s.area_cap += s.cap() * dt
        s.last = now
    def _shift_clock(s):
        while True:
            yield s.env.timeout(15)
            s._dispatch()
    def request(s, prio):
        ev = s.env.event(); s.seq += 1
        ent = [prio, s.seq, ev, False]
        heapq.heappush(s.q, ent); s._dispatch()
        return ent
    def cancel(s, ent): ent[3] = True
    def _dispatch(s):
        s._tick()
        while s.q and s.in_use < s.cap():
            ent = heapq.heappop(s.q)
            if ent[3]: continue
            s.in_use += 1; ent[2].succeed()
    def release(s):
        s._tick(); s.in_use -= 1; s._dispatch()

def _prio(cls):
    """Priority from a class label: its leading number ('3', '3-blue' -> 3.0); 0 when there is none. Lower is served first."""
    m = re.match(r"\s*(-?\d+(\.\d+)?)", str(cls)); return float(m.group(1)) if m else 0.0

def _draw(q, rnd):
    return float(np.interp(rnd.random(), QGRID, q))

def _lognormal(mean, cv, rnd):
    s2 = math.log(1 + cv * cv); return rnd.lognormvariate(math.log(mean) - s2 / 2, math.sqrt(s2))

def simulate(P, days=60, warmup_days=10, seed=1, capacity_delta=None, arrival_mult=1.0, hold_mult=None, gap_mult=None, capacity_set=None):
    """Run the process. capacity_set: {pool: n or [[start_hour, n], ...]} sets a resource level; capacity_delta: {pool: +/-n};
    arrival_mult scales arrivals; hold_mult: {pool: factor} for external holds;
    gap_mult: {activity: factor} scales the time taken to reach that activity (e.g. faster lab results)."""
    from .spec import with_capacity, releases
    import random
    rnd = random.Random(seed); nprnd = np.random.default_rng(seed)
    env = simpy.Environment(); rows = []
    hold_mult = hold_mult or {}; gap_mult = gap_mult or {}
    t0 = pd.Timestamp(P["arrival"]["t0"]); hour0 = t0.hour + t0.minute / 60.0; dow0 = t0.dayofweek
    pools = {}
    for p in with_capacity(P["pools"], capacity_set, capacity_delta):
        p["hour0"] = hour0
        pools[p["name"]] = Pool(env, p)
    acq = {p["acquire"]: pools[p["name"]] for p in P["pools"]}
    rel = {}
    for p in P["pools"]:
        for r in releases(p): rel.setdefault(r, []).append(pools[p["name"]])
    warm = warmup_days * 1440.0
    for pl in pools.values(): pl.log_from = warm
    prof = np.array(P["arrival"]["profile"]) * arrival_mult
    lam_max = prof.max()
    classes = list(P["class_mix"]); cw = [P["class_mix"][c] for c in classes]
    trans, patience, abandon = P["trans"], P["patience"], P["abandon"]
    visits = {}
    for v in P.get("visits", []): visits.setdefault(v["before"], []).append(v)
    cls_of = {}

    def emit(cid, act, now): rows.append((cid, act, now))
    def hold_external(pl, name, hold):
        d = _lognormal(hold["mean"] * hold_mult.get(name, 1.0), hold["cv"], rnd); yield env.timeout(d); pl.release()
    def prefill():
        for p in P["pools"]:
            h = p.get("hold")
            if h and h.get("initial_occupancy"):
                pl = pools[p["name"]]
                for _ in range(int(round(h["initial_occupancy"] * pl.cap()))):
                    pl.in_use += 1
                    env.process(hold_external(pl, p["name"], h))
    def case(cid, cls):
        tab = trans.get(cls) or trans["__all__"]
        node = P["start"]; emit(cid, node, env.now); held = []
        while True:
            outs = tab.get(node)
            if not outs: break
            r = rnd.random(); acc = 0.0; pick = outs[-1]
            for o in outs:
                acc += o[1]
                if r <= acc: pick = o; break
            nxt, _, q = pick
            pl = acq.get(nxt)
            if pl is not None:
                ent = pl.request(_prio(cls) if pl.p.get("priority_by_class") else 0.0)
                mean_pat = patience.get(pl.p["name"]) if node in abandon else None
                if mean_pat:
                    tau = rnd.expovariate(1.0 / mean_pat)
                    yield ent[2] | env.timeout(tau)
                    if not ent[2].triggered:
                        pl.cancel(ent); emit(cid, abandon[node], env.now); break
                else:
                    yield ent[2]
                emit(cid, nxt, env.now); held.append(pl)
                h = pl.p.get("hold")
                if h and not releases(pl.p):
                    held.remove(pl); env.process(hold_external(pl, pl.p["name"], h))
            else:
                gap = _draw(q, rnd) * gap_mult.get(nxt, 1.0)
                vs = [v for v in visits.get(nxt, []) if not v.get("class_match") or re.search(v["class_match"], str(cls))]
                if vs:   # hidden work: part of this gap is service by a limited resource that the log does not show
                    k = vs[0]["scale"]; vp = pools[vs[0]["pool"]]
                    yield env.timeout((1 - k) * gap)
                    ent = vp.request(_prio(cls) if vp.p.get("priority_by_class") else 0.0); yield ent[2]
                    yield env.timeout(k * gap); vp.release()
                else:
                    yield env.timeout(gap)
                emit(cid, nxt, env.now)
            for pl2 in rel.get(nxt, []):
                if pl2 in held: held.remove(pl2); pl2.release()
            node = nxt
        for pl2 in held: pl2.release()
    def arrivals():
        n = 0; end = warm + days * 1440.0
        while env.now < end and lam_max > 0:
            yield env.timeout(rnd.expovariate(lam_max))
            hrs = env.now / 60.0 + hour0
            d = int((dow0 + hrs // 24) % 7); h = int(hrs % 24)
            if rnd.random() <= prof[d][h] / lam_max:
                n += 1; cid = f"S{seed}-{n}"; c = rnd.choices(classes, cw)[0]; cls_of[cid] = c
                env.process(case(cid, c))
    prefill(); env.process(arrivals()); env.run(until=warm + days * 1440.0 + 3 * 1440.0)
    df = pd.DataFrame(rows, columns=["case_id", "activity", "t"])
    first = df.groupby("case_id").t.first(); keep = first[first >= warm].index
    df = df[df.case_id.isin(keep)].copy()
    done = df.groupby("case_id").activity.last().isin(set(P["terminals"]))     # drop cases still in progress at the end
    df = df[df.case_id.isin(done[done].index)].copy()
    df["t"] -= warm
    df["ts"] = pd.Timestamp(P["arrival"]["t0"]) + pd.to_timedelta(df.t + warm, unit="m")
    df["resource"] = pd.NA
    df = df.sort_values(["case_id", "t"], kind="mergesort").reset_index(drop=True)
    util = {n: (pl.area_busy / pl.area_cap if pl.area_cap else float("nan")) for n, pl in pools.items()}
    cls_series = pd.Series({c: cls_of[c] for c in df.case_id.unique()})
    df.attrs["t0"] = pd.Timestamp(P["arrival"]["t0"]) + pd.Timedelta(minutes=warm)
    return df, cls_series, util
