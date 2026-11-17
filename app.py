"""Streamlit app: pick an event log, see its process map, check the simulation against the log, then try changes.

Run from the project root:  .venv/bin/streamlit run app.py
"""
import glob, html, json, os
import numpy as np, pandas as pd, streamlit as st
import altair as alt
from engine.log import load_log
from engine.spec import draft_spec, load_spec
from engine.fit import fit
from engine.discover import merged_map, mermaid_merged, merge_rules
from engine.validate import validate
from engine.calibrate import validation_loop
from engine.scenarios import spec_for
from engine.sim import simulate
from engine.kpis import summarize, samples, quit_ends
from engine.report import pdf_figure, pretty, STATUS, mermaid_html, build as build_report, add_section
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Process simulation", layout="wide", page_icon="🏥")
BASE_C, SCEN_C, BAD = "#9aa5b1", "#0f766e", "#b42318"

st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1400px}
h1{font-size:1.75rem!important;margin-bottom:0!important}
.chips span{display:inline-block;background:#fff;border:1px solid #e2e6ea;border-radius:14px;padding:2px 10px;margin:0 6px 6px 0;font-size:.82rem;color:#3d4852}
.step{background:#eef6f5;border-left:4px solid #0f766e;border-radius:6px;padding:9px 14px;margin:4px 0 16px;color:#1f3b38;font-size:.92rem}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:10px;margin:4px 0 14px}
.card{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:11px 14px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.card .lab{font-size:.78rem;color:#5f6b76;line-height:1.25;min-height:2.1em}
.card .val{font-size:1.5rem;font-weight:650;color:#16212b;font-variant-numeric:tabular-nums;line-height:1.3}
.card .val small{font-size:.85rem;font-weight:500;color:#5f6b76}
.card .from{font-size:.8rem;color:#5f6b76;font-variant-numeric:tabular-nums}
.card .delta{display:inline-block;margin-top:4px;font-size:.78rem;font-weight:600;border-radius:8px;padding:1px 7px}
.good{color:#1e7b45;background:#dcf1e3}.bad{color:#b42318;background:#fbe1de}.flat{color:#5f6b76;background:#eef1f3}
.card.pass{border-top:4px solid #1e7b45}.card.noise{border-top:4px solid #c58a00}.card.fail{border-top:4px solid #b42318}.card.info{border-top:4px solid #9aa5b1}
.panel-h{font-weight:650;font-size:.95rem;margin:10px 0 2px;color:#16212b}
.hint{font-size:.78rem;color:#5f6b76;margin:-2px 0 4px}
div[data-testid="stForm"]{background:#fff;border:1px solid #e2e6ea;border-radius:10px}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers
def case_list():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "cases", "*", "case.json"))):
        c = json.load(open(f)); c["dir"] = os.path.dirname(f); out[c.get("title", os.path.basename(c["dir"]))] = c
    return out

@st.cache_data(show_spinner="Reading the log")
def load_case(log_path, cases_path, class_col, spec_path, params_path):
    log, classes = load_log(log_path, cases_path=cases_path, class_col=class_col)
    spec = load_spec(spec_path) if spec_path else draft_spec(log)
    P = json.load(open(params_path)) if params_path and os.path.exists(params_path) else fit(log, classes, spec)
    return log, classes, spec, P

@st.cache_data(show_spinner=False)
def run_reps(P_json, spec_json, kw_json, reps, days, warmup, seed0=201):
    """Replications of one setting: KPI table per run, per-case samples (for histograms), simulated utilization."""
    P, spec, kw = json.loads(P_json), json.loads(spec_json), json.loads(kw_json)
    sp = spec_for(spec, kw); terminals = set(P["terminals"]); out = []
    for s in range(seed0, seed0 + reps):
        lg, cls, util = simulate(P, days, warmup, s, **kw)
        out.append((summarize(lg, cls, sp, terminals), samples(lg, cls, sp, terminals), util))
    return out

@st.cache_data(show_spinner=False)
def observed(_log, _classes, spec_json, terms, case_key):
    return summarize(_log, _classes, json.loads(spec_json), set(terms))

def per_run(runs):
    """KPI values of each replication: index (kpi, segment, stat), one column per run."""
    return pd.concat([r[0].set_index(["kpi", "segment", "stat"]).value for r in runs], axis=1)

def fmt(v, unit):
    if v is None or pd.isna(v): return "–"
    if unit == "min":
        return f"{v / 60:.1f}<small> h</small>" if abs(v) >= 120 else f"{v:.0f}<small> min</small>" if abs(v) >= 10 else f"{v:.1f}<small> min</small>"
    if unit == "pct": return f"{v:.1f}<small>%</small>"
    return f"{v:,.2f}"

def plain(s): return s.replace("<small>", "").replace("</small>", "")

def cards(items):
    st.markdown('<div class="cards">' + "".join(
        f'<div class="card {i.get("cls", "")}"><div class="lab">{html.escape(i["label"])}</div><div class="val">{i["value"]}</div>'
        + (f'<div class="from">{i["from"]}</div>' if i.get("from") else "")
        + (f'<span class="delta {i.get("tone", "flat")}">{i["delta"]}</span>' if i.get("delta") else "") + "</div>" for i in items) + "</div>",
        unsafe_allow_html=True)

def step(text): st.markdown(f'<div class="step">{text}</div>', unsafe_allow_html=True)

def shift_label(sh, i):
    return f"{sh[i][0]:02d}–{sh[(i + 1) % len(sh)][0]:02d}"

def headline(case, spec, quits):
    """Measures shown as cards: case.json "headline" if given, else time in system, abandonment, and every resource wait."""
    if case.get("headline"): return [tuple(h) for h in case["headline"]]
    h = [("Time in system, average", "cycle_time_min", "All", "mean", "min"), ("Time in system, 90th percentile", "cycle_time_min", "All", "p90", "min")]
    h += [(f"Share who {q.lower()}" if q.lower().startswith("left") else f"Share ending '{q}'", "outcome_share_pct", q, "value", "pct") for q in sorted(quits)]
    h += [(f"Wait for {p['name']}, average", f"wait_for_{p['name']}_min", "All", "mean", "min") for p in spec["pools"]]
    return h

# ---------------------------------------------------------------- sidebar
cases = case_list()
with st.sidebar:
    st.markdown("### 🏥 Process simulation")
    choice = st.selectbox("Event log", list(cases) + ["Upload a log…"], key="case")
    if choice == "Upload a log…":
        up = st.file_uploader("Event log CSV (case_id, activity, timestamp, optional resource)", type="csv", key="upload")
        if not up: st.info("Upload a CSV to start."); st.stop()
        path = os.path.join(ROOT, ".uploads"); os.makedirs(path, exist_ok=True); fp = os.path.join(path, up.name)
        open(fp, "wb").write(up.getvalue())
        c = dict(title=up.name, dir=path, log=up.name, days=60, warmup_days=10)
    else:
        c = cases[choice]
    j = lambda k: os.path.join(c["dir"], c[k]) if c.get(k) else None
    log, classes, spec, P0 = load_case(j("log"), j("cases"), c.get("class_col"), j("spec"), j("params"))
    st.markdown("**How to use**\n1. **Process map**: see how cases flow and where they wait.\n2. **Validate**: check the simulation reproduces the log.\n"
                "3. **What-if**: change staff, spaces, demand or length of stay and compare with today.")
    with st.expander("Simulation settings"):
        reps = st.number_input("Replications", 2, 20, 4, key="reps", help="Independent runs per setting; more runs = less noise, slower.")
        days = st.number_input("Days per run", 20, 365, int(c.get("days", 60)), step=10, key="days")
        warmup = st.number_input("Warm-up days (discarded)", 0, 90, int(c.get("warmup_days", 10)), key="warmup",
                                 help="The model starts empty; these first days are thrown away.")
        tol = st.slider("Validation target", 0.01, 0.25, 0.05, 0.01, format="%.2f", key="tol",
                        help="A measure passes when the simulation is within this share of the log (5% is the project target).")
    st.caption(f"Runs: {reps} × {days} days (+{warmup} warm-up). Validation target ±{tol:.0%}.")

key = f"P::{choice}"
if key not in st.session_state: st.session_state[key] = P0
P = st.session_state[key]; terminals = set(P["terminals"]); quits = quit_ends(log, spec, terminals)
obs = observed(log, classes, json.dumps(spec), sorted(terminals), choice).set_index(["kpi", "segment", "stat"]).value
og = lambda k, seg="All", s="value": float(obs.get((k, seg, s), np.nan))
HL = headline(c, spec, quits)

st.markdown(f"# {html.escape(c.get('title', 'Process simulation'))}")
st.markdown(f'<div class="chips"><span>{log.case_id.nunique():,} cases</span><span>{log.t.max() / 1440:.0f} days of data</span>'
            f'<span>{og("cases_per_day"):.0f} cases / day</span><span>{log.activity.nunique()} activities</span>'
            f'<span>{len(spec["pools"])} resources</span></div>', unsafe_allow_html=True)
tab_map, tab_val, tab_sc, tab_rep = st.tabs(["1 · Process map", "2 · Validate", "3 · What-if", "4 · Report"])
rkey = f"rep::{choice}"

def make_report(val, summ, logs, kind):
    """Build the report from a validation run in the app and keep only the HTML (the simulated logs are large)."""
    with st.spinner("Building the report"):
        note = (f"{kind} run in the app on {datetime.datetime.now():%Y-%m-%d %H:%M}: {reps} replications × {days} days "
                f"(+{warmup} warm-up), target ±{tol:.0%}.")
        st.session_state[rkey] = (build_report(None, c.get("title", choice), log, classes, spec, P, val, summ, logs, note=note), note)

# ---------------------------------------------------------------- 1 process map
with tab_map:
    step("<b>What the data shows today.</b> Boxes are steps (median and 90th-percentile time in the step); arrows show how many cases "
         "take that path and the average wait before the next step (<b style='color:#b7791f'>amber</b> over 15 min, <b style='color:#b42318'>red</b> over 60). "
         "Zoom with + / −, double-click, or pinch; drag to pan.")
    cards([dict(label=l, value=fmt(og(k, seg, s), u)) for l, k, seg, s, u in HL])
    mm = merged_map(log); big = len(mm["nodes"]) > 8
    st.iframe(mermaid_html(mermaid_merged(mm), height=560 if big else 320), height=620 if big else 380)
    st.markdown('<div class="panel-h">Queues and resources in the log</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"resource": p["name"],
                                "capacity": str(p["capacity"]) if not p.get("shifts") else
                                " / ".join(f"{c_} ({shift_label(sorted(p['shifts']), i)})" for i, (h, c_) in enumerate(sorted(p["shifts"]))),
                                "avg number waiting": og(f"queue_avg_{p['name']}"), "longest queue": og(f"queue_max_{p['name']}"),
                                "avg wait (min)": og(f"wait_for_{p['name']}_min", s="mean"), "90th pct wait (min)": og(f"wait_for_{p['name']}_min", s="p90"),
                                "utilization %": og(f"utilization_{p['name']}_pct")} for p in spec["pools"]]).round(1), hide_index=True, width="stretch")
    with st.expander("Every path and its wait"):
        e = pd.DataFrame(mm["edges"]); e = e[e.n >= 0.01 * mm["n_cases"]]
        st.dataframe(e.rename(columns={"proc": "from", "next": "to", "n": "cases", "mean_wait": "avg wait (min)", "median_wait": "median wait (min)",
                                       "p90_wait": "90th pct wait (min)"}).round(1), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 2 validate
with tab_val:
    step(f"<b>Can we trust the simulation?</b> It runs {reps} times and every measure is computed by the same code on the log and on the "
         f"simulation. Green: within ±{tol:.0%}. Amber: outside it but within the spread between runs (needs more runs). Red: a real gap. "
         "<i>Validation loop</i> also re-tunes what the log cannot show (e.g. physician time per patient, ward stay) and re-checks.")
    c1, c2, _ = st.columns([1, 1.3, 4])
    run_val = c1.button("Run validation", type="primary", key="run_val", width="stretch")
    run_loop = c2.button("Run validation loop", key="run_loop", width="stretch", help="Takes a few minutes on the ED cases.")
    vkey = f"val::{choice}::{reps}::{days}::{warmup}::{tol}"
    if run_loop:
        with st.spinner("Validation loop: tuning and re-validating (a few minutes)"):
            Pn, val, summ, hist, vlogs = validation_loop(P, spec, log, classes, tol, reps=reps, days=days, warmup_days=warmup, log=lambda m: None)
        st.session_state[key] = P = Pn; summ["loop_history"] = hist; st.session_state[vkey] = (val, summ, hist)
        make_report(val, summ, vlogs, "Validation loop"); del vlogs
    elif run_val:
        with st.spinner(f"Running {reps} replications"):
            val, summ, vlogs = validate(P, spec, log, classes, reps, days, warmup, tol=tol, keep_logs=True)
        st.session_state[vkey] = (val, summ, None)
        make_report(val, summ, vlogs, "Validation"); del vlogs
    if vkey in st.session_state:
        val, summ, vhist = st.session_state[vkey]; n = int((val.status != "INFO").sum())
        cnt = lambda s_: int((val.status == s_).sum())
        cards([dict(label=f"Within ±{tol:.0%} (target)", value=f"{cnt('PASS')}<small> / {n}</small>", cls="pass"),
               dict(label="Within run-to-run noise", value=f"{cnt('NOISE')}<small> / {n}</small>", cls="noise"),
               dict(label="Real gaps", value=f"{cnt('FAIL')}<small> / {n}</small>", cls="fail"),
               dict(label="Within ±10% · ±20%", value=f"{summ['within_10pct']} · {summ['within_20pct']}<small> / {summ['checks']}</small>", cls="info")])
        if vhist:
            st.markdown('<div class="panel-h">Validation loop rounds</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"round": h["round"], f"within ±{tol:.0%}": f"{h['passed']}/{h['checks']}",
                                        **{k: round(v, 3) for k, v in h["tuned"].items()}} for h in vhist]), hide_index=True)
        only = st.segmented_control("Show", ["PASS", "NOISE", "FAIL", "INFO"], default=["PASS", "NOISE", "FAIL", "INFO"],
                                    selection_mode="multi", key="val_filter", format_func=lambda s_: STATUS[s_][2]) or []
        v = val[val.status.isin(only)].assign(measure=lambda d: d.kpi.map(pretty), difference=lambda d: 100 * d.rel_err)
        v = v[["measure", "segment", "stat", "observed", "sim_mean", "sim_sd", "difference", "status"]].rename(
            columns={"observed": "log", "sim_mean": "simulation", "sim_sd": "± between runs", "difference": "difference %"})
        st.dataframe(v.style.apply(lambda col: [f"color:{STATUS.get(x, STATUS['INFO'])[0]};background-color:{STATUS.get(x, STATUS['INFO'])[1]};font-weight:600"
                                                for x in col], subset=["status"])
                     .format({"log": "{:,.2f}", "simulation": "{:,.2f}", "± between runs": "{:,.2f}", "difference %": "{:+.1f}%"}),
                     hide_index=True, width="stretch", height=420)
        st.markdown('<div class="panel-h">Distributions: log vs simulation</div>', unsafe_allow_html=True)
        obs_s = samples(log, classes, spec, terminals)
        sim_runs = run_reps(json.dumps(P), json.dumps(spec), "{}", reps, days, warmup, seed0=101)
        pick = st.multiselect("Measures", list(obs_s), [k for k in obs_s if k.endswith("|All")][:4], key="val_hist",
                              format_func=lambda k: f"{pretty(k.split('|')[0])}, {k.split('|')[1]}")
        cols = st.columns(2)
        for i, k in enumerate(pick):
            kpi, seg = k.split("|")
            cols[i % 2].pyplot(pdf_figure(obs_s[k], [r[1].get(k, []) for r in sim_runs], f"{pretty(kpi)}, {seg}"))
    else:
        st.info("Click **Run validation** to compare the simulation with the log.")

# ---------------------------------------------------------------- 3 what-if
with tab_sc:
    step("<b>Try changes and compare them with today.</b> Set any number of levers on the left (staff by shift, beds and chairs, demand, "
         "length of stay) and click <b>Run what-if</b>. Cards show today → with your changes. A change is marked "
         "<b>clear</b> only when it is bigger than the run-to-run noise; otherwise it says <i>within noise</i>.")
    ver = st.session_state.setdefault(f"lv::{choice}", 0)
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        with st.form(f"levers_{choice}_{ver}", border=True):
            st.markdown('<div class="panel-h">Staff and spaces</div>', unsafe_allow_html=True)
            cap_set, changed = {}, []
            for p in spec["pools"]:
                n_ = p["name"]
                if p.get("shifts"):
                    sh = sorted(p["shifts"]); st.markdown(f'<div class="hint"><b>{html.escape(n_)}</b> on duty, by shift</div>', unsafe_allow_html=True)
                    cols = st.columns(len(sh)); new = []
                    for i, ((h, c_), col) in enumerate(zip(sh, cols)):
                        new.append([h, int(col.number_input(shift_label(sh, i), 0, 500, int(c_), key=f"v{ver}_{choice}_{n_}_{h}"))])
                    if new != [list(x) for x in sh]:
                        cap_set[n_] = new; changed.append(f"{n_} {'/'.join(str(x) for _, x in sh)}→{'/'.join(str(x) for _, x in new)}")
                else:
                    v_ = int(st.number_input(n_, 0, 5000, int(p["capacity"]), key=f"v{ver}_{choice}_{n_}"))
                    if v_ != int(p["capacity"]): cap_set[n_] = v_; changed.append(f"{n_} {p['capacity']}→{v_}")
            st.markdown('<div class="panel-h">Demand</div>', unsafe_allow_html=True)
            arr = st.slider("Arrivals", -40, 60, 0, 5, format="%d%%", key=f"v{ver}_{choice}_arr")
            hold = {}
            holds = [p["name"] for p in spec["pools"] if p.get("hold")]
            if holds:
                st.markdown('<div class="panel-h">Length of stay</div>', unsafe_allow_html=True)
                for hn in holds:
                    pct = st.slider(f"{hn} stay", -40, 40, 0, 5, format="%d%%", key=f"v{ver}_{choice}_hold_{hn}")
                    if pct: hold[hn] = 1 + pct / 100; changed.append(f"{hn} stay {pct:+d}%")
            with st.expander("Step durations (advanced)"):
                rules = merge_rules(log.activity.unique()); ends = {r[2]: r[0] for r in rules}
                acq = {p["acquire"] for p in spec["pools"]}
                targets = sorted({b for tab in P["trans"].values() for outs in tab.values() for b, _, q in outs if q is not None and b not in acq})
                names = {t: (f"{ends[t]} (time in step)" if t in ends else f"time before '{t}'") for t in targets}
                gstep = st.selectbox("Step", ["(none)"] + targets, format_func=lambda t: names.get(t, t), key=f"v{ver}_{choice}_gstep")
                gpct = st.slider("Change in duration", -60, 60, 0, 5, format="%d%%", key=f"v{ver}_{choice}_gpct")
            go = st.form_submit_button("Run what-if", type="primary", width="stretch")
        if st.button("Reset levers to today", key=f"reset_{choice}", width="stretch"):
            st.session_state[f"lv::{choice}"] = ver + 1; st.rerun()

    kw = {}
    if cap_set: kw["capacity_set"] = cap_set
    if arr: kw["arrival_mult"] = 1 + arr / 100; changed.append(f"arrivals {arr:+d}%")
    if hold: kw["hold_mult"] = hold
    if gstep != "(none)" and gpct: kw["gap_mult"] = {gstep: 1 + gpct / 100}; changed.append(f"{names[gstep]} {gpct:+d}%")
    hkey = f"hist::{choice}"
    if go:
        if not kw: st.toast("Nothing changed yet: adjust at least one lever.")
        else:
            with st.spinner(f"Simulating today and your changes ({reps} runs each)"):
                base = run_reps(json.dumps(P), json.dumps(spec), "{}", reps, days, warmup)
                scen = run_reps(json.dumps(P), json.dumps(spec), json.dumps(kw), reps, days, warmup)
            st.session_state.setdefault(hkey, []).append(dict(label="; ".join(changed), base=base, scen=scen, kw=kw))
            st.session_state[f"sel_{choice}"] = len(st.session_state[hkey]) - 1
    hist = st.session_state.get(hkey, [])

    with right:
        if not hist:
            st.markdown('<div class="panel-h">Today (from the data)</div>', unsafe_allow_html=True)
            cards([dict(label=l, value=fmt(og(k, seg, s), u)) for l, k, seg, s, u in HL])
            st.info("Set the levers and click **Run what-if**. Each run simulates today and your changes side by side "
                    "(about 10–30 seconds on the ED cases).")
        else:
            labels = [h["label"] for h in hist]
            sel = st.selectbox("Showing", range(len(hist)), format_func=lambda i: f"Run {i + 1}: {labels[i]}", key=f"sel_{choice}")
            h = hist[sel]; B, S = per_run(h["base"]), per_run(h["scen"]); nb, ns = B.shape[1], S.shape[1]
            items = []
            for lab, k, seg, s, u in HL:
                ix = (k, seg, s)
                if ix not in B.index: continue
                b = B.loc[ix].mean(); sv = S.loc[ix].mean() if ix in S.index else np.nan
                se = np.sqrt(B.loc[ix].var(ddof=1) / nb + (S.loc[ix].var(ddof=1) / ns if ix in S.index else 0))
                d = sv - b; rel = 100 * d / b if abs(b) > 1e-9 else np.nan
                clear = abs(d) > 2 * se and abs(d) > 1e-6
                tone = "flat" if not clear else ("good" if d < 0 else "bad")
                dtxt = "–" if np.isnan(rel) else f"{'▼' if d < 0 else '▲'} {abs(rel):.0f}% · {'clear' if clear else 'within noise'}"
                items.append({"label": lab, "value": fmt(sv, u), "from": f"today {plain(fmt(b, u))}", "delta": dtxt, "tone": tone})
            cards(items)

            ub = {k: 100 * np.mean([r[2][k] for r in h["base"]]) for k in h["base"][0][2]}
            us = {k: 100 * np.mean([r[2][k] for r in h["scen"]]) for k in h["scen"][0][2]}
            udf = pd.DataFrame([{"resource": k, "setting": "today", "utilization %": ub[k]} for k in ub] +
                               [{"resource": k, "setting": "with changes", "utilization %": us.get(k, np.nan)} for k in ub])
            wk = lambda p: (f"wait_for_{p['name']}_min", "All", "mean")
            wdf = pd.DataFrame([{"resource": p["name"], "setting": lab_, "avg wait (min)": float(X.loc[wk(p)].mean())}
                                for p in spec["pools"] for lab_, X in [("today", B), ("with changes", S)] if wk(p) in X.index])
            colour = alt.Color("setting:N", scale=alt.Scale(domain=["today", "with changes"], range=[BASE_C, SCEN_C]),
                               legend=alt.Legend(orient="top", title=None))
            def bars(df, x, title, rule=None):
                ch = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
                    y=alt.Y("resource:N", title=None, sort=None), yOffset="setting:N", x=alt.X(f"{x}:Q", title=x), color=colour,
                    tooltip=["resource", "setting", alt.Tooltip(f"{x}:Q", format=".1f")])
                if rule is not None:
                    ch = ch + alt.Chart(pd.DataFrame({"x": [rule]})).mark_rule(color=BAD, strokeDash=[4, 3]).encode(x="x:Q")
                return ch.properties(title=title, height=34 * df.resource.nunique() + 30)
            g1, g2 = st.columns(2)
            g1.altair_chart(bars(udf, "utilization %", "Utilization (dashed line: 80% target)", 80), width="stretch")
            g2.altair_chart(bars(wdf, "avg wait (min)", "Average wait for each resource"), width="stretch")

            with st.expander("All measures"):
                rows = []
                for ix in B.index:
                    k, seg, s = ix
                    if seg != "All" and not k.startswith("outcome_share"): continue
                    b = B.loc[ix].mean(); sv = S.loc[ix].mean() if ix in S.index else np.nan
                    rows.append({"measure": pretty(k), "segment": seg, "stat": s, "today": b, "with changes": sv,
                                 "change %": 100 * (sv - b) / b if abs(b) > 1e-9 else np.nan})
                st.dataframe(pd.DataFrame(rows).style.format({"today": "{:,.2f}", "with changes": "{:,.2f}", "change %": "{:+.1f}%"}),
                             hide_index=True, width="stretch", height=360)
            with st.expander("Distribution of one measure"):
                k = st.selectbox("Measure", list(h["base"][0][1]), key=f"dist_{choice}",
                                 format_func=lambda k: f"{pretty(k.split('|')[0])}, {k.split('|')[1]}")
                st.pyplot(pdf_figure(np.concatenate([r[1].get(k, []) for r in h["base"]]), [r[1].get(k, []) for r in h["scen"]],
                                     f"{pretty(k.split('|')[0])}, {k.split('|')[1]}", labels=("today", "with changes")))

    if hist:
        st.markdown('<div class="panel-h">All what-if runs</div>', unsafe_allow_html=True)
        def row(name, runs):
            M = per_run(runs).mean(axis=1)
            return {"run": name, **{lab: round(float(M.get((k, seg, s), np.nan)), 1) for lab, k, seg, s, u in HL}}
        tab = [row("Today", hist[0]["base"])] + [row(f"{i + 1}: {x['label']}", x["scen"]) for i, x in enumerate(hist)]
        st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
        if st.button("Clear what-if runs", key=f"clear_{choice}"):
            st.session_state[hkey] = []; st.session_state.pop(f"sel_{choice}", None); st.rerun()

# ---------------------------------------------------------------- 4 report
def whatif_section(hist):
    """The what-if runs as a report section: today and each run, headline measures with the change from today."""
    def u(v, unit): return plain(fmt(v, unit))
    head = "".join(f"<th>{html.escape(l)}</th>" for l, *_ in HL)
    rows = []
    for i, x in enumerate([None] + hist):
        runs = hist[0]["base"] if x is None else x["scen"]; M = per_run(runs).mean(axis=1)
        Bm = per_run(hist[0]["base"] if x is None else x["base"]).mean(axis=1)
        cells = []
        for l, k, seg, s, unit in HL:
            v, b = float(M.get((k, seg, s), np.nan)), float(Bm.get((k, seg, s), np.nan))
            ch = "" if x is None or not abs(b) > 1e-9 else f" <span class=sd>({100 * (v - b) / b:+.0f}%)</span>"
            cells.append(f"<td>{u(v, unit)}{ch}</td>")
        name = "Today (simulated)" if x is None else f"{i}: {html.escape(x['label'])}"
        rows.append(f"<tr><td>{name}</td>{''.join(cells)}</tr>")
    return (f"<h2>What-if runs</h2><p class=muted>Each run changes the listed levers together and is compared with today, both simulated with "
            f"{len(hist[0]['base'])} replications. Percentages are the change from today; small changes in measures that swing a lot "
            f"between runs (boarding) may be noise.</p><div class=wrap><table><thead><tr><th>Run</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")

with tab_rep:
    step("<b>One page to share.</b> The report combines the process map, queues, validation (colour-coded against the target), "
         "the validation loop, your what-if runs, and log-vs-simulation histograms. Running <i>Validate</i> in this session rebuilds it "
         "from your run; otherwise the saved report for this case is shown. Download it to email or attach it.")
    saved = os.path.join(c["dir"], "results", "report.html")
    if rkey in st.session_state:
        doc, note = st.session_state[rkey]; src = note
    elif os.path.exists(saved):
        doc = open(saved).read(); src = (f"Saved report (built offline with the validation loop; file last updated "
                                         f"{datetime.datetime.fromtimestamp(os.path.getmtime(saved)):%Y-%m-%d %H:%M}). Run Validate to rebuild it from this session.")
    else:
        doc = None
    if doc is None:
        st.info("No report yet for this log. Run **2 · Validate** first.")
    else:
        hist = st.session_state.get(hkey, [])
        inc = st.toggle(f"Include my what-if runs ({len(hist)})", value=bool(hist), disabled=not hist, key=f"inc_{choice}_{len(hist)}")
        if inc and hist: doc = add_section(doc, whatif_section(hist))
        a, b = st.columns([3, 1])
        a.caption(src)
        fname = f"{os.path.basename(c['dir'])}_report_{datetime.date.today():%Y%m%d}.html"
        b.download_button("Download report (HTML)", doc, file_name=fname, mime="text/html", type="primary", width="stretch")
        st.iframe(doc, height=1100)
