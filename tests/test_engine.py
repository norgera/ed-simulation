import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "tools"))
import pytest
from make_demo_logs import single_server, coffee_shop
from engine.log import load_log
from engine.discover import discover
from engine.spec import draft_spec
from engine.fit import fit
from engine.validate import validate

@pytest.fixture(scope="module")
def ss(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "ss.csv"; single_server(str(p), days=45)
    log, cls = load_log(str(p)); spec = draft_spec(log); P = fit(log, cls, spec)
    return log, cls, spec, P

def test_discovers_structure(ss):
    log, cls, spec, P = ss
    pm = discover(log)
    assert pm["start"] == "Arrival" and set(pm["terminals"]) == {"Service end"}
    assert [(p["name"], p["capacity"]) for p in spec["pools"]] == [("server", 1)]

def test_matches_textbook_queue(ss):
    """Arrivals 20/h, service 30/h, one server: exact answers are 66.7% busy, 4.0 min mean wait."""
    log, cls, spec, P = ss
    val, summ = validate(P, spec, log, cls, reps=3, days=40)
    row = lambda k, s: val[(val.kpi == k) & (val.stat == s)].iloc[0]
    assert abs(row("utilization_server_pct", "value").sim_mean - 66.7) < 2.0
    assert abs(row("wait_for_server_min", "mean").sim_mean - 4.0) < 0.5
    assert abs(row("little_L_over_lambdaW", "value").sim_mean - 1.0) < 0.01

def test_coffee_shop_pools_and_patience(tmp_path):
    p = tmp_path / "cs.csv"; coffee_shop(str(p), days=45)
    log, cls = load_log(str(p)); spec = draft_spec(log); P = fit(log, cls, spec)
    caps = {x["name"]: x["capacity"] for x in spec["pools"]}
    assert caps == {"cashier": 1, "barista": 2}
    assert abs(P["patience"]["cashier"] - 6.0) < 1.0           # customers walk away after ~6 minutes on average
    val, summ = validate(P, spec, log, cls, reps=3, days=40)
    assert summ["within_10pct"] >= 0.85 * summ["checks"]

def test_column_mapping(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("visit,step,when\nA,in,2025-01-01 00:00:00\nA,out,2025-01-01 00:05:00\nB,in,2025-01-01 00:01:00\nB,out,2025-01-01 00:03:00\n")
    log, _ = load_log(str(p), case_col="visit", activity_col="step", time_col="when")
    assert discover(log)["n_cases"] == 2
    with pytest.raises(ValueError):
        load_log(str(p))

# ---- added with the November meeting items: merged map, queue statistics, resource levels, release lists, zones
from engine.discover import merge_rules, merged_map
from engine.spec import hold_intervals, with_capacity
from engine.sim import simulate

def test_merge_rules_pairs_start_and_end():
    rules = merge_rules(["Arrive", "Order start", "Order end", "Prep start", "Prep end", "Pickup", "Lab ordered/collected", "Lab result"])
    assert ("Order", "Order start", "Order end") in rules and ("Prep", "Prep start", "Prep end") in rules
    assert ("Lab", "Lab ordered/collected", "Lab result") in rules

def test_merged_map_coffee(tmp_path):
    p = tmp_path / "cs.csv"; coffee_shop(str(p), days=10)
    log, _ = load_log(str(p)); mm = merged_map(log)
    names = {n["proc"] for n in mm["nodes"]}
    assert {"Order", "Prep"} <= names and "Order start" not in names
    wait = next(e for e in mm["edges"] if e["proc"] == "Order" and e["next"] == "Prep")
    assert wait["mean_wait"] > 0          # customers wait for a barista between ordering and preparation

def test_queue_length_matches_mm1(ss):
    """M/M/1 with rho = 2/3: mean number waiting Lq = rho^2 / (1 - rho) = 1.33."""
    log, cls, spec, P = ss
    val, _ = validate(P, spec, log, cls, reps=3, days=40)
    row = val[val.kpi == "queue_avg_server"].iloc[0]
    assert abs(row.observed - 1.333) < 0.25 and abs(row.sim_mean - 1.333) < 0.25
    assert val[val.kpi == "queue_max_server"].status.iloc[0] == "INFO"

def test_capacity_set_levels_and_shifts(ss):
    log, cls, spec, P = ss
    pools = with_capacity(spec["pools"], {"server": [[0, 1], [12, 2]]})
    assert pools[0]["shifts"] == [[0, 1], [12, 2]] and pools[0]["capacity"] == 2
    _, _, u1 = simulate(P, 20, 5, 3); _, _, u2 = simulate(P, 20, 5, 3, capacity_set={"server": 2})
    assert u2["server"] < 0.6 * u1["server"]          # twice the servers, about half the utilization

def test_release_list_takes_first_release(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("case_id,activity,timestamp\nA,Chair,2025-01-01 00:00\nA,Moved,2025-01-01 00:10\nA,Leave,2025-01-01 00:30\n"
                 "B,Chair,2025-01-01 00:00\nB,Leave,2025-01-01 00:20\n")
    log, _ = load_log(str(p)); iv = hold_intervals(log, "Chair", ["Moved", "Leave"])
    assert iv.loc["A"].r - iv.loc["A"].a == 10 and iv.loc["B"].r - iv.loc["B"].a == 20

def test_zone_ed_branch_is_not_abandonment():
    """In the zone ED a discharge follows the decision, and the decision also precedes a queue (admitted chair patients
    wait for a stretcher). The discharge must stay a normal path, not be read as patients giving up."""
    import importlib.util, pathlib, numpy as np
    root = pathlib.Path(__file__).parents[1]
    spec_ = importlib.util.spec_from_file_location("gen", root / "tools" / "gen_synthetic_ed.py"); gen = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(gen)
    ed, start, wu = gen.run(30, 5, zones=True)
    import pandas as pd
    ev = pd.DataFrame(ed.events, columns=["case_id", "activity", "timestamp", "resource"])
    cs = pd.DataFrame(ed.cases)
    assert {"Green chair assigned", "Blue chair assigned", "Yellow stretcher assigned", "Red stretcher assigned"} <= set(ev.activity)
    assert set(cs[cs.zone == "green"].ctas) <= {4, 5}
    d = pathlib.Path(__file__).parent / "_zone_tmp.csv"; ev.to_csv(d, index=False)
    try:
        log, _ = load_log(str(d)); zspec = json_load(root / "cases" / "ed_zones" / "spec.json")
        Pz = fit(log, None, zspec)
        assert "Disposition decision" not in Pz["abandon"]
        assert any(b == "Departure from ED" for b, _, _ in Pz["trans"]["__all__"]["Disposition decision"])
    finally:
        d.unlink()

def json_load(p):
    import json; return json.load(open(p))
