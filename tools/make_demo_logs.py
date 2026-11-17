"""Make two small event logs with known structure, using code that is independent of the engine.
  single_server.csv : arrivals 20/h, one server serving 30/h. Textbook answer: 67% busy, 4 min wait, 6 min in system.
  coffee_shop.csv   : morning/lunch peaks, one cashier, two baristas, impatient customers walk away."""
import random, sys, simpy, pandas as pd
from datetime import datetime, timedelta

def write(rows, path):
    df = pd.DataFrame(rows, columns=["case_id", "activity", "timestamp", "resource", "kind"])
    df = df.sort_values(["case_id", "timestamp"], kind="mergesort"); df.to_csv(path, index=False)
    return df

def single_server(path, days=90, lam=20, mu=30, seed=1):
    rnd = random.Random(seed); env = simpy.Environment(); srv = simpy.Resource(env, 1); rows = []
    t0 = datetime(2025, 1, 6)
    ts = lambda m: (t0 + timedelta(minutes=m)).isoformat(sep=" ")
    def cust(i):
        cid = f"C{i}"; rows.append((cid, "Arrival", ts(env.now), "", "")); 
        with srv.request() as r:
            yield r; rows.append((cid, "Service start", ts(env.now), "server", ""))
            yield env.timeout(rnd.expovariate(mu / 60)); rows.append((cid, "Service end", ts(env.now), "server", ""))
    def arr():
        i = 0
        while env.now < days * 1440:
            yield env.timeout(rnd.expovariate(lam / 60)); i += 1; env.process(cust(i))
    env.process(arr()); env.run(until=days * 1440 + 600)
    return write(rows, path)

def coffee_shop(path, days=120, seed=2):
    rnd = random.Random(seed); env = simpy.Environment()
    cashier = simpy.Resource(env, 1); barista = simpy.Resource(env, 2); rows = []
    t0 = datetime(2025, 1, 6)
    ts = lambda m: (t0 + timedelta(minutes=m)).isoformat(sep=" ")
    rate = lambda h: 0 if h < 7 or h >= 18 else (55 if 7 <= h < 10 else 40 if 11 <= h < 14 else 22)   # customers per hour
    def logn(mean, cv):
        import math; s2 = math.log(1 + cv * cv); return rnd.lognormvariate(math.log(mean) - s2 / 2, math.sqrt(s2))
    def cust(i):
        cid = f"K{i}"; rows.append((cid, "Arrive", ts(env.now), "", ""))
        with cashier.request() as r:
            res = yield r | env.timeout(rnd.expovariate(1 / 6.0))          # patience: mean 6 minutes
            if r not in res:
                rows.append((cid, "Walked away", ts(env.now), "", "")); return
            rows.append((cid, "Order start", ts(env.now), "cashier", "")); yield env.timeout(logn(0.9, 0.5))
            rows.append((cid, "Order end", ts(env.now), "cashier", ""))
        with barista.request() as r:
            yield r; rows.append((cid, "Prep start", ts(env.now), "barista", "")); yield env.timeout(logn(2.6, 0.6))
            rows.append((cid, "Prep end", ts(env.now), "barista", ""))
        yield env.timeout(rnd.uniform(0.1, 0.8)); rows.append((cid, "Pickup", ts(env.now), "", ""))
    def arr():
        i = 0
        while env.now < days * 1440:
            h = (env.now / 60) % 24; lam = rate(h)
            if lam == 0: yield env.timeout(5); continue
            yield env.timeout(rnd.expovariate(80 / 60))                   # thinning with max rate 80/h
            if rnd.random() < rate((env.now / 60) % 24) / 80: i += 1; env.process(cust(i))
    env.process(arr()); env.run(until=days * 1440 + 600)
    return write(rows, path)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    a = single_server(f"{out}/single_server.csv"); b = coffee_shop(f"{out}/coffee_shop.csv")
    print(len(a), "events;", len(b), "events")
