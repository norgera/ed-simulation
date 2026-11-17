"""Self-contained HTML report (charts embedded as base64 PNG)."""
import base64, io, html
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .scenarios import TARGET_UTIL

def _png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return '<img src="data:image/png;base64,%s">' % base64.b64encode(b.getvalue()).decode()

def _ecdf(ax, x, label, **kw):
    x = np.sort(np.asarray(x.dropna())); ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=label, **kw)

def fig_validation(obs, sims):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (col, ttl, xmax) in zip(axs, [("door_to_PIA_min", "Door-to-PIA (min)", 240), ("LOS_min", "ED length of stay (min)", 900), ("boarding_min", "Boarding time, admitted (min)", 1500)]):
        o = obs[col] if col != "boarding_min" else obs.loc[obs.group == "Admitted", col]
        s = pd.concat([(c[col] if col != "boarding_min" else c.loc[c.group == "Admitted", col]) for c in sims])
        _ecdf(ax, o, "Observed", lw=2.2, color="#1f4e79"); _ecdf(ax, s, "Simulated", lw=1.8, ls="--", color="#d9822b")
        ax.set_xlim(0, xmax); ax.set_title(ttl, fontsize=10); ax.set_ylabel("cumulative share"); ax.grid(alpha=.25)
    axs[0].legend(frameon=False); fig.tight_layout(); return _png(fig)

def fig_util(df):
    cols = {"util_ed_beds": "ED beds", "util_physician": "Physicians", "util_ward": "Inpatient beds", "util_di": "Imaging", "util_triage": "Triage nurses"}
    fig, ax = plt.subplots(figsize=(14, 4.6)); n = len(df); w = 0.8 / len(cols)
    pal = ["#1f4e79", "#d9822b", "#5b8c5a", "#8c6bb1", "#999999"]
    for i, (c, lab) in enumerate(cols.items()):
        ax.bar(np.arange(n) + i * w, 100 * df[c], w, label=lab, color=pal[i])
    ax.axhline(100 * TARGET_UTIL, color="crimson", ls="--", lw=1.2); ax.text(n - .4, 100 * TARGET_UTIL + 1, "80% target", color="crimson", ha="right", fontsize=8)
    ax.set_xticks(np.arange(n) + 0.4 - w / 2); import textwrap; ax.set_xticklabels([textwrap.fill(s, 16) for s in df.index], fontsize=6.3, rotation=0)
    ax.set_ylabel("mean utilization %"); ax.set_ylim(0, 105); ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.12)); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); return _png(fig)

def fig_kpis(df):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (c, t) in zip(axs, [("pia_p90", "Door-to-PIA, 90th pct (min)"), ("los_p90_all", "LOS, 90th pct (min)"), ("lwbs_pct", "Left without being seen (%)")]):
        ax.barh(range(len(df)), df[c], color=["#1f4e79"] + ["#d9822b"] * (len(df) - 1)); ax.set_yticks(range(len(df)))
        ax.set_yticklabels([s[:34] for s in df.index] if ax is axs[0] else [], fontsize=7); ax.invert_yaxis(); ax.set_title(t, fontsize=10); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); return _png(fig)

def _tbl(df, fmt=None, cls=""):
    fmt = fmt or {}
    h = "<table class='%s'><tr>%s</tr>" % (cls, "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns)))
    for i, r in df.iterrows():
        h += "<tr><td>%s</td>%s</tr>" % (html.escape(str(i)), "".join(f"<td>{(fmt.get(c, '{:.1f}').format(v) if isinstance(v, (int, float, np.floating)) else html.escape(str(v)))}</td>" for c, v in r.items()))
    return h + "</table>"

def build(path, P, cfg, val, ks, obs, sims, scen):
    ok = int(val.ok.sum()); n = len(val)
    v = val.copy(); v["observed"] = v.observed.round(2); v["simulated"] = v.simulated.round(2)
    v["rel_err"] = (100 * v.rel_err).round(1).astype(str) + "%"; v["tol"] = v.tol.round(2); v["ok"] = np.where(v.ok, "PASS", "FAIL")
    v = v.drop(columns="sim_sd").set_index("kpi")
    vt = _tbl(v, {c: "{}" for c in v.columns}).replace("<td>PASS</td>", "<td class='ok'>PASS</td>").replace("<td>FAIL</td>", "<td class='bad'>FAIL</td>")
    s = scen.copy()
    show = s[["pia_median", "pia_p90", "los_median_non", "los_p90_all", "boarding_mean", "lwbs_pct", "throughput_per_day", "util_ed_beds", "util_physician", "util_ward", "pct_time_beds_full"]].copy()
    for c in ["util_ed_beds", "util_physician", "util_ward", "pct_time_beds_full"]: show[c] = 100 * show[c]
    show.columns = ["PIA median", "PIA p90", "LOS med (non-adm)", "LOS p90", "Boarding mean", "LWBS %", "Visits/day out", "ED bed util %", "Physician util %", "Ward util %", "% time ED full"]
    show.index.name = "Scenario"
    cal = P["calibration"]
    css = """body{font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1b1f23;background:#fff}
    h1{font-size:24px}h2{margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:4px}table{border-collapse:collapse;font-size:12px;margin:10px 0;width:100%}
    td,th{border-bottom:1px solid #e3e3e3;padding:3px 7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f3f5f8}
    .ok{color:#0a7a3d;font-weight:600}.bad{color:#c0392b;font-weight:600}.note{background:#fff7e6;border-left:4px solid #e0a030;padding:8px 12px;margin:12px 0}img{max-width:100%}"""
    doc = f"""<!doctype html><meta charset=utf-8><title>ED simulation MVP report</title><style>{css}</style>
    <h1>ED discrete-event simulation: MVP report</h1>
    <div class=note><b>Data are synthetic.</b> The event log was produced by an assumed-parameter generator, so absolute numbers say nothing about a real hospital. What this report demonstrates is the pipeline: log &rarr; fitted parameters &rarr; DES model &rarr; validation &rarr; scenarios.</div>
    <h2>1. Validation of the baseline model ({ok}/{n} checks pass)</h2>
    <p>Same KPI code on observed log and on {len(sims)} simulated replications (120 days each). KS distance between observed and simulated distributions: door-to-PIA {ks['door_to_PIA_min']:.3f}, LOS {ks['LOS_min']:.3f}.
    Calibrated (unobservable) parameters: reassessment scale k by CTAS = {', '.join(f'{v:.2f}' for v in cal['k_reassess'].values())}; inpatient LOS mean = {cal['ward_los_mean_min']/1440:.2f} days.</p>
    {fig_validation(obs, sims)}{vt}
    <h2>2. Scenario analysis (mean of replications)</h2>
    <p>Staffing/bed levers are changed in the model only; fitted service times are untouched. Times in minutes.</p>
    {_tbl(show)}{fig_kpis(scen)}
    <h2>3. Utilization vs 80% resilience target</h2>
    {fig_util(scen)}
    <h2>4. Discovered process map (top directly-follows transitions)</h2>
    {_tbl(pd.DataFrame(P['process_map']).head(20).set_index('activity'), {'count': '{:.0f}'})}
    <h2>5. Limitations</h2><ul>
    <li>Physician count, shifts, bed counts, scanner counts are <i>inputs</i> from the hospital; an event log cannot reveal them. Here: physicians {cfg['physician']}, triage nurses {cfg['triage']}, ED beds {cfg['ed_beds']}, inpatient beds {cfg['ward_beds']}.</li>
    <li>Reassessment+nursing work and inpatient LOS are not observed in an ED log; they are calibrated (k, ward LOS), so scenarios that change them are less certain than those that change observed quantities.</li>
    <li>ED nurses, lab stations, porters are not capacity-constrained (no data); lab/imaging report time is a pure delay.</li>
    <li>Single-site, single ED area; no fast-track/resus zones, shift-change effects or ward discharge time-of-day.</li></ul>"""
    open(path, "w").write(doc)
