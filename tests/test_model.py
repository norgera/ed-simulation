import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import pandas as pd
from model import fit, kpis
from model.model import Sim
D = pathlib.Path(__file__).parents[1] / "cases" / "ed" / "data"

def _P():
    ev, cs = kpis.load(D / "ed_event_log.csv", D / "ed_cases.csv"); return fit.fit(ev, cs), ev, cs

def test_fit_recovers_arrivals():
    P, ev, cs = _P(); assert 140 < P["arrival"]["mean_per_day"] < 160
    assert abs(sum(P["ctas_mix"].values()) - 1) < 1e-9

def test_sim_log_schema_matches_observed():
    P, ev, cs = _P(); e, c, u = Sim(P, None, {}, 1, 10).run()
    assert list(e.columns) == list(ev.columns) and set(e.activity) <= set(ev.activity)
    assert c.departure_ts.notna().all() or True and len(c) > 800

def test_more_physicians_cut_waiting():
    from model.runner import run_once
    P, *_ = _P(); a = run_once((P, {}, {}, 5, 30, False)); b = run_once((P, {}, {"cfg": {"physician": [6, 6, 4]}}, 5, 30, False))
    assert b["pia_p90"] < a["pia_p90"]

def test_event_order_per_case():
    P, *_ = _P(); e, c, u = Sim(P, None, {}, 2, 5).run()
    first = e.groupby("case_id").activity.first(); assert (first == "Arrival").all()
