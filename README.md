# ED simulation capstone (CS4470Y)

Event log -> process map -> fitted resource model -> SimPy simulation -> validation vs observed KPIs -> what-if scenarios.

    pip install -r requirements.txt
    streamlit run app.py                        # app: 1 process map, 2 validate, 3 what-if (any number of levers at once), 4 report (view + download)
    python -m pytest tests                      # engine tests
    python -m engine run --log cases/coffee_shop/data/coffee_shop.csv --out cases/coffee_shop/results
    python -m engine whatif --params cases/ed/results/params.json --spec cases/ed/spec.json \
        --scenarios cases/ed/scenarios.json --out cases/ed/results

## Layout
| Path | What |
|---|---|
| `app.py` | Scenario screen (Streamlit): pick a case or upload a log, see the merged process map and queues, validate against a tolerance (5% default) with colour-coded results, then set any combination of levers (staff per shift, beds/chairs, arrivals, length of stay, a step's duration) and compare with today: headline cards say whether each change is bigger than the run-to-run noise. Headline measures come from `headline` in `case.json` (list of [label, kpi, segment, stat, unit]) or a generic default. The Report tab shows the saved `results/report.html`, or rebuilds it from a validation run in the app, optionally with the what-if runs, and downloads it as one HTML file |
| `engine/` | The generic engine (reads any event log). See `engine/README.md` |
| `cases/` | One folder per log: `data/`, `case.json` (settings for the app), `spec.json` (ED), `scenarios.json`, `results/` (with `report.html`). Cases: `single_server`, `coffee_shop`, `ed` (beds first), `ed_zones` (zone-based triage) |
| `model/` | The ED-specific model (hand-built, earlier version) kept as a cross-check; see `model/README.md`. Its outputs are in `cases/ed/reference_results/` |
| `tools/` | Synthetic log generators and shared KPI code |
| `reports/` | `ed_simulation_report.html`, `engine_examples_report.html` |
| `tests/` | Engine tests (`test_engine.py`, `test_model.py`) |

All logs here are synthetic; the engine is meant to be rerun on a real log.

## Zone-based ED (`cases/ed_zones`)
`python tools/gen_synthetic_ed.py --zones --out cases/ed_zones/data` makes a log in which triage sends each patient to a zone
instead of straight to a bed: red (resuscitation), yellow (acute, cannot stay vertical: stretcher), blue (vertical, results pending: chair)
and green (fast track: chair, own provider). Admitted chair patients move to a yellow stretcher to board. The zone rules, space counts and
staffing are placeholders (see `ZONES` and `ZCAP` in the generator) until the supervisor's ideal ED process map arrives. Run it with:

    python -m engine run --log cases/ed_zones/data/ed_event_log.csv --cases cases/ed_zones/data/ed_cases.csv \
        --class-col ctas_zone --spec cases/ed_zones/spec.json --loop --warmup 30 --days 120 --reps 6 --out cases/ed_zones/results
