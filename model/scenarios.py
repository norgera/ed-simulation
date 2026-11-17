"""Scenario definitions (levers) and comparison vs baseline. Staffing/bed levers live under 'cfg'; others are multipliers."""
import pandas as pd
from .runner import replicate

SCENARIOS = {
    "Baseline": {},
    "+1 physician per shift": {"cfg": {"physician": [5, 5, 3]}},
    "+4 ED beds (30->34)": {"cfg": {"ed_beds": 34}},
    "+6 inpatient beds (93->99)": {"cfg": {"ward_beds": 99}},
    "Inpatient LOS -10% (faster ward discharge)": {"ward_los_mult": 0.90},
    "Lab & imaging turnaround -25%": {"lab_mult": 0.75, "di_mult": 0.75},
    "ED discharge process -30%": {"discharge_mult": 0.70},
    "Demand +10%": {"arrival_mult": 1.10},
    "Combined: +1 phys, ward LOS -10%, lab/DI -25%": {"cfg": {"physician": [5, 5, 3]}, "ward_los_mult": 0.90, "lab_mult": 0.75, "di_mult": 0.75},
    "Combined package under demand +10%": {"cfg": {"physician": [5, 5, 3]}, "ward_los_mult": 0.90, "lab_mult": 0.75, "di_mult": 0.75, "arrival_mult": 1.10},
}

TARGET_UTIL = 0.80
UTIL_COLS = ["util_ed_beds", "util_physician", "util_triage", "util_di", "util_ward"]

def run_scenarios(P, cfg, scenarios=None, seeds=(201, 202, 203, 204), days=120, workers=2, log=print):
    scenarios = scenarios or SCENARIOS; rows = []
    for name, sc in scenarios.items():
        log(f"  scenario: {name}")
        reps = pd.DataFrame(replicate(P, cfg, sc, seeds, days, workers))
        r = reps.mean().to_dict(); r["scenario"] = name
        for c in ["pia_median", "pia_p90", "los_median_non", "los_p90_all", "lwbs_pct", "boarding_mean", "util_physician", "util_ward"]:
            r[c + "_sd"] = float(reps[c].std())
        rows.append(r)
    df = pd.DataFrame(rows).set_index("scenario")
    return df

def delta_table(df):
    base = df.iloc[0]
    cols = ["pia_median", "pia_p90", "los_median_non", "los_p90_all", "boarding_mean", "lwbs_pct", "throughput_per_day"]
    d = df[cols].copy()
    for c in cols: d[c + "_delta_%"] = 100 * (df[c] - base[c]) / base[c]
    return d
