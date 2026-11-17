# engine: a generic event-log-to-simulation engine

Give it any event log (case id, activity, timestamp, optional resource). It discovers the process map, drafts
which resources limit the process, fits arrivals, branching and step durations, simulates the process (SimPy),
and checks the simulation against the log with the same KPI code on both. It has no emergency-department
logic in it. The ED log runs through it with one small spec file.

## Quick start
```
pip install -r requirements.txt
python tools/make_demo_logs.py cases    # regenerate the demo logs (see tools/make_demo_logs.py)
python -m engine run --log cases/coffee_shop/data/coffee_shop.csv --out cases/coffee_shop/results
python -m pytest -q tests
```
Your own columns: `--case-col visit --activity-col step --time-col when --resource-col who`.
Classes (for example acuity level): `--class-col ctas`, plus `--cases cases.csv` if the column is in a separate table.
A resource spec you have edited: `--spec my_spec.json`. Tolerance (default 5%): `--tol 0.10`.
Validation loop (re-tune what the log cannot show until checks are within the tolerance or stop improving): `--loop`.
Warm-up days discarded from each run: `--warmup 30` (use about 30 when a nearly full ward is modelled; it takes weeks to settle).

## What it does, in order
1. **Load** the log into one schema and sort events within each case.
2. **Discover** activities, the directly-follows graph, start and end activities. Writes `process_map.mmd` (Mermaid), in which
   a start event and its end event ("Order start", "Order end") are one process box with its median and p90 duration, and each
   arrow shows the average wait before the next step. `process_map_raw.mmd` keeps every event as its own box.
3. **Draft a resource spec.** A resource label on two consecutive events of a case is a hold (seize then release).
   Capacity is the peak number of overlapping holds in the log. The draft is saved as `spec_drafted.json`
   with notes on anything it could not work out.
4. **Fit** arrival rate by hour and weekday, class mix, transition probabilities, step durations (empirical
   distributions, per class), and customer patience (censored-exponential estimate) where cases leave while waiting.
5. **Simulate** with queues, priorities, time-varying capacity and abandonment.
6. **Validate** every KPI on log and simulation, report PASS / NOISE / FAIL, and how many checks fall within
   5%, 10% and 20%. Time and percentage measures have a small absolute floor (0.1 minute, 0.5 points).
   Queue measures per resource: time-weighted average number waiting (scored) and the longest queue (shown, not scored,
   because it grows with the length of the period).
7. **Report** `report.html`: merged process map, queues, colour-coded validation table, and histograms (probability
   densities) of log vs simulation for time in system and the wait for each resource.

## The spec (what a log cannot always show)
A log shows what happened. It does not always show which resources were busy, so the spec lists:
- `pools`: `name`, `acquire` event, `release` event, `capacity`, optional `shifts` as `[start_hour, capacity]`,
  `priority_by_class`. A pool with no release event can use `hold` (mean, cv, initial_occupancy): an external stay such as a ward bed.
- `visits`: hidden work before an event (a share of the observed gap is service by a pool). `--calibrate <pool>` finds that share.
  Optional `class_match` (a regular expression on the class label) limits an entry to some classes, e.g. `"green$"`.
- `abandon` (optional): the ending event(s) that mean a case gave up while waiting, e.g. `["Left without being seen"]`.
  Without it, every ending except the most common one is treated as giving up.
- `release` may be a list of events: the unit is freed by whichever comes first (a chair freed by a move to a stretcher or by departure).

## Results
| Log | What the engine found by itself | Checks within 5% / 10% / 20% |
|---|---|---|
| single server, 20 arrivals/h, 30 services/h | 1 server; model 67.6% busy and 4.3 min wait, log 67.1% and 4.1 (exact answers 66.7% and 4.0) | 11/12, 12/12, 12/12 |
| coffee shop | cashier x1, baristas x2; walk-away patience 5.97 min (true 6.0) | 20/20, 20/20, 20/20 |
| synthetic ED log, 55k visits | needed a spec for beds, ward and shifts, plus one calibration | 14/42, 22/42, 32/42 |

The ED-specific model (`ed_mvp`) scores 8/28, 16/28, 23/28 on its own check set. The two sets differ, so this is a rough
comparison and not a controlled one. The ED log is synthetic: results show the method works and nothing about a real hospital.

## Limits
- Branching is first-order: the next step depends on the current step and the class, not on earlier history.
- Capacity drafted from the log is the peak overlap. When staffing varies by shift, supply `shifts`.
- A resource held across several events needs the spec (the ED bed is held from "ED bed assigned" to "Departure"), as does one with no release event (ward).
- Hidden work is a modelling assumption, calibrated to one KPI. Check it with more than that KPI.
- Batching, preemption, and several resources needed at once are not modelled.
- Measures driven by a nearly full resource (boarding for a ward at 93-97%) swing a lot between runs; they need long runs and
  more replications before a 5% comparison means anything. The NOISE status marks those.
- Not built yet: AnyLogic export. What-if: `python -m engine whatif ...`, the scenario screen (`app.py`), or
  `simulate(..., capacity_set={...}, capacity_delta={...}, arrival_mult=..., gap_mult={...}, hold_mult={...})`.
