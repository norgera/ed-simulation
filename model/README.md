# model - ED discrete-event simulation MVP

Event log -> fitted parameters -> SimPy DES -> validation vs observed KPIs -> scenario analysis -> HTML report.

    # run from the project root (pip install -r requirements.txt once)
    python -m model all --log cases/ed/data/ed_event_log.csv --cases cases/ed/data/ed_cases.csv --out cases/ed/reference_results   # ~3 min
    python -m model all --quick ...        # ~1 min smoke run
    python -m pytest tests/test_model.py

Inputs: `ed_event_log.csv` (case_id, activity, timestamp, resource) and `ed_cases.csv` (case_id, ctas, outcome, admitted, arrival_ts, departure_ts).
Optional `--cfg cfg.json` overrides resource config (`triage`, `physician` = [day, evening, night], `ed_beds`, `di`, `ward_beds`).

## Pipeline (module -> what it does)
| Module | Role |
|---|---|
| `kpis.py` | Per-case table + KPIs (mean/median/p90 of door-to-triage, door-to-PIA, boarding, LOS by CTAS and admitted/non-admitted; bed occupancy). **Same code for observed and simulated logs.** |
| `fit.py` | Process map (directly-follows), arrival rate (base x hour x weekday), CTAS mix, branch probabilities (lab/DI/consult/admit), lognormal service/delay fits per CTAS, LWBS patience (censored-exponential MLE) |
| `model.py` | SimPy model: triage -> ED bed -> physician initial assessment -> lab/imaging (parallel) -> consult -> reassessment -> disposition -> discharge or boarding for an inpatient bed. Priority by CTAS, shift-based staffing, LWBS, inpatient ward with background occupancy. Emits the same event-log schema. |
| `calibrate.py` | Fits what a log cannot show: reassessment scale `k` and mean inpatient LOS (bisection to match observed decision interval and boarding time) |
| `validate.py` | Replications vs observed with tolerance checks (mean 20%, median 15%, p90 25%), KS distances |
| `scenarios.py` | Lever-based scenarios (staffing, beds, demand, service-time multipliers, ward LOS, admit rate, CTAS mix) vs baseline, utilization vs 80% target |
| `report.py` | Self-contained HTML report |

Scenario levers: `cfg` {physician, triage, ed_beds, di, ward_beds} and multipliers `arrival_mult, pia_mult, lab_mult, di_mult, consult_mult, reassess_mult, discharge_mult, admit_mult, ward_los_mult`, plus `ctas_mix`. Add your own in `scenarios.SCENARIOS`.

## Status on the synthetic log
24/28 validation checks pass (LOS and boarding well matched; KS 0.02). Door-to-PIA is ~30% too slow (overall mean/median/p90, CTAS-4 median).
Likely cause: reassessment/nursing work is unobserved and modelled as one calibrated physician stage, which overstates physician load.
Do not read the simulated physician utilization (~91%) as validated - there is no observed ground truth for it.

## Known limits / next steps
1. Fit on a real log (MIMIC-IV-ED/MIMICEL lacks physician/lab/DI timestamps; a Canadian hospital log is needed for PIA).
2. Staffing rosters, bed counts and inpatient occupancy (ADT) are inputs, not inferable from an ED log.
3. Boarding is very sensitive near ward saturation (93-98% load): small ward changes swing boarding a lot, so scenarios use 6 replications and report spread (`*_sd` in scenario_results.csv).
4. Add ED nurse pool, lab stations, fast-track/resus zones; time-of-day ward discharges.
5. AnyLogic: `results/params_calibrated.json` is the parameter contract for a parallel AnyLogic model; compare both against the log with the same `kpis.py` outputs.
6. Replace the greedy scenario list with a search (e.g. smallest staffing/bed change that brings physician and bed utilization <=80% without hurting throughput).
