"""Self-contained HTML report for one log: merged process map, validation (colour-coded against the tolerance),
queue statistics, and distributions of observed vs simulated values drawn as histograms (probability densities)."""
import base64, html, io
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .discover import merged_map, mermaid_merged
from .kpis import samples

OBS, SIM = "#1f2d3a", "#2b7bb9"
STATUS = {"PASS": ("#1e7b45", "#dcf1e3", "within tolerance"), "NOISE": ("#9a6700", "#fbefd0", "within noise"),
          "FAIL": ("#b42318", "#fbe1de", "miss"), "INFO": ("#5a6865", "#eef1f1", "shown, not scored"), "MISSING": ("#5a6865", "#eef1f1", "missing")}

def mermaid_html(code, height=560):
    """Mermaid diagram with zoom and pan: + / − / reset buttons, double-click to zoom in, drag to pan,
    pinch or Ctrl/⌘ + scroll to zoom (a plain scroll still scrolls the page)."""
    return f"""<div class=mmwrap style="position:relative;background:#fff"><pre class=mermaid>{html.escape(code)}</pre>
<div style="font:12px sans-serif;color:#5a6865;padding:2px 4px">Zoom: + / − buttons, double-click, pinch or Ctrl/⌘ + scroll. Drag to pan.</div></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
<script>
mermaid.initialize({{startOnLoad:false,securityLevel:"strict",theme:"base",flowchart:{{curve:"basis"}}}});
mermaid.run({{querySelector:".mermaid"}}).then(function(){{
  document.querySelectorAll(".mermaid svg").forEach(function(svg){{
    svg.style.maxWidth="none"; svg.setAttribute("width","100%"); svg.setAttribute("height","{height}px");
    var pz=svgPanZoom(svg,{{zoomEnabled:true,controlIconsEnabled:true,mouseWheelZoomEnabled:false,dblClickZoomEnabled:true,
                           fit:true,center:true,minZoom:0.2,maxZoom:30,zoomScaleSensitivity:0.3}});
    svg.addEventListener("wheel",function(e){{
      if(!(e.ctrlKey||e.metaKey)) return; e.preventDefault();
      var r=svg.getBoundingClientRect(); pz.zoomAtPointBy(Math.exp(-e.deltaY/150),{{x:e.clientX-r.left,y:e.clientY-r.top}});
    }},{{passive:false}});
    window.addEventListener("resize",function(){{pz.resize();pz.fit();pz.center();}});
  }});
}});
</script>"""

def pdf_figure(obs, sims, title, bins=40, ax=None, labels=("log", "simulation")):
    """Overlaid histograms (densities) of observed and simulated values. Values above the 99th percentile are grouped
    in the last bin so a long tail does not squash the picture."""
    obs = np.asarray(obs, float); obs = obs[~np.isnan(obs)]
    sim = np.concatenate([np.asarray(s, float) for s in sims]) if len(sims) else np.array([])
    sim = sim[~np.isnan(sim)]
    own = ax is None
    if own: fig, ax = plt.subplots(figsize=(5.2, 3.0))
    both = np.concatenate([obs, sim])
    if len(both) == 0: return None
    hi = max(np.quantile(both, .99), 1e-6); edges = np.linspace(0, hi, bins + 1)
    for x, lab, col, kw in [(obs, labels[0], OBS, dict(histtype="stepfilled", alpha=.28)), (sim, labels[1], SIM, dict(histtype="step", lw=2))]:
        if len(x): ax.hist(np.clip(x, 0, hi), bins=edges, density=True, color=col, label=lab, **kw)
    z_o = 100 * np.mean(obs <= 0.05) if len(obs) else np.nan; z_s = 100 * np.mean(sim <= 0.05) if len(sim) else np.nan
    sub = (f"no wait: {labels[0]} {z_o:.0f}%, {labels[1]} {z_s:.0f}%" if "wait" in title and z_o > 5
           else f"n = {len(obs):,} {labels[0]}, {len(sim):,} {labels[1]}")
    ax.set_title(f"{title}\n{sub}", fontsize=9, loc="left"); ax.set_xlabel("minutes", fontsize=8); ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=.2); ax.legend(fontsize=7, frameon=False)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    if own: fig.tight_layout(); return fig

def _png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}">'

def pretty(kpi):
    k = kpi.replace("_min", " (min)").replace("_pct", " (%)")
    for a, b in [("wait_for_", "wait for "), ("utilization_", "utilization "), ("cycle_time", "time in system"),
                 ("queue_avg_", "avg number waiting: "), ("queue_max_", "longest queue: "), ("cases_per_day", "cases per day"),
                 ("outcome_share", "share ending"), ("little_L_over_lambdaW", "Little's law check (L / λW)")]:
        k = k.replace(a, b)
    return k

def queue_table(val, spec):
    """One row per pool: average and longest queue, mean wait, utilization; log vs simulation."""
    v = val.set_index(["kpi", "stat"]); rows = []
    get = lambda k, s, c: float(v.loc[(k, s), c]) if (k, s) in v.index else np.nan
    for p in spec["pools"]:
        n = p["name"]
        rows.append({"resource": n, "capacity": p["capacity"] if not p.get("shifts") else " / ".join(str(c) for _, c in sorted(p["shifts"])),
                     **{f"{lab} {src}": get(k, s, col) for lab, k, s in [("avg waiting", f"queue_avg_{n}", "value"), ("longest queue", f"queue_max_{n}", "value"),
                                                                       ("mean wait (min)", f"wait_for_{n}_min", "mean"), ("utilization %", f"utilization_{n}_pct", "value")]
                        for src, col in [("log", "observed"), ("sim", "sim_mean")]}})
    return pd.DataFrame(rows)

def status_table(val, tol):
    out = ["<table><thead><tr><th>Measure</th><th>Segment</th><th>Stat</th><th>Log</th><th>Simulation</th><th>Difference</th><th>Status</th></tr></thead><tbody>"]
    for r in val.itertuples():
        fg, bg, lab = STATUS.get(r.status, STATUS["INFO"])
        sd = f" <span class=sd>± {r.sim_sd:.2f}</span>" if not pd.isna(r.sim_sd) else ""
        diff = f"{100 * r.rel_err:+.1f}%" if abs(r.observed) > 1e-9 else f"{r.rel_err:+.2f}"
        out.append(f"<tr><td>{html.escape(pretty(r.kpi))}</td><td>{html.escape(str(r.segment))}</td><td>{r.stat}</td><td>{r.observed:,.2f}</td>"
                   f"<td>{r.sim_mean:,.2f}{sd}</td><td>{diff}</td><td><span class=pill style='color:{fg};background:{bg}'>{r.status.lower()}</span></td></tr>")
    return "".join(out) + "</tbody></table>"

def _df_html(df, fmt="{:,.2f}"):
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr></thead><tbody>"
    for _, r in df.iterrows():
        h += "<tr>" + "".join(f"<td>{fmt.format(x) if isinstance(x, (float, np.floating)) and not pd.isna(x) else html.escape(str(x))}</td>" for x in r) + "</tr>"
    return h + "</tbody></table>"

def build(path, name, log, classes, spec, P, val, summ, sim_logs):
    tol = summ.get("tol", 0.05); terminals = set(P["terminals"])
    mm = merged_map(log); chk = summ["checks"]; n_pass = int((val.status == "PASS").sum())
    obs_s = samples(log, classes, spec, terminals)
    sim_s = [samples(l, c, spec, terminals) for l, c in sim_logs]
    figs = []
    for key in obs_s:
        if key.startswith("cycle_time_min|ends") and len(obs_s[key]) < 0.02 * len(obs_s["cycle_time_min|All"]): continue
        kpi, seg = key.split("|"); figs.append(_png(pdf_figure(obs_s[key], [s.get(key, []) for s in sim_s], f"{pretty(kpi)}, {seg}")))
    nodes = pd.DataFrame(mm["nodes"]); edges = pd.DataFrame(mm["edges"])
    nodes = nodes.rename(columns={"proc": "step", "n": "cases", "median_min": "median (min)", "p90_min": "p90 (min)"}).drop(columns="merged")
    edges = edges[edges.n >= 0.01 * mm["n_cases"]].rename(columns={"proc": "from", "next": "to", "n": "cases", "mean_wait": "avg wait (min)",
                                                                   "median_wait": "median wait (min)", "p90_wait": "p90 wait (min)"})
    util = summ.get("sim_pool_utilization", {})
    hist = summ.get("loop_history")
    loop_html = "" if not hist else "<h2>Validation loop</h2><p class=muted>Each round re-tunes what the log cannot show, then validates again.</p>" + \
        _df_html(pd.DataFrame([{"round": h["round"], "checks within tolerance": f"{h['passed']}/{h['checks']}", "within 10%": h["within_10pct"],
                                 "tuned values": ", ".join(f"{k} = {v:.3g}" for k, v in h["tuned"].items())} for h in hist]), "{}")
    css = """:root{--ink:#14201e;--muted:#5a6865;--rule:#d8dfdd;--bg:#fbfcfc}
body{font:14px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);max-width:1120px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-top:2px solid var(--ink);padding-top:10px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;margin:8px 0}th,td{padding:5px 8px;border-bottom:1px solid var(--rule);text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{font-weight:600;color:var(--muted);font-size:12px}
.pill{padding:1px 8px;border-radius:9px;font-size:12px;font-weight:600}.sd{color:var(--muted);font-size:11px}
.tiles{display:flex;gap:12px;flex-wrap:wrap}.tile{border:1px solid var(--rule);border-radius:6px;padding:10px 14px;background:#fff;min-width:150px}
.tile b{font-size:22px;display:block}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}.grid img{width:100%;background:#fff;border:1px solid var(--rule);border-radius:6px}
.map{background:#fff;border:1px solid var(--rule);border-radius:6px;padding:10px;overflow-x:auto}.wrap{overflow-x:auto}"""
    tiles = "".join(f"<div class=tile><b>{v}</b>{k}</div>" for k, v in [
        (f"checks within {tol:.0%}", f"{n_pass}/{chk}"), ("within 10%", f"{summ['within_10pct']}/{chk}"), ("within 20%", f"{summ['within_20pct']}/{chk}"),
        ("cases in the log", f"{log.case_id.nunique():,}"), ("days in the log", f"{log.t.max() / 1440:.0f}")])
    doc = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>{html.escape(name)} simulation report</title>
<style>{css}</style></head><body>
<h1>{html.escape(name)}: simulation report</h1>
<p class=muted>Event log → process map → fitted simulation → validation. Every measure is computed by the same code on the log and on the simulation.</p>
<div class=tiles>{tiles}</div>
<h2>Process map</h2><p class=muted>Start and end events of the same step are merged into one box, which shows the median and 90th percentile time in that step.
Arrows show how many cases take that path and the average wait before the next step (amber over 15 min, red over 60 min).</p>
<div class=map>{mermaid_html(mermaid_merged(mm))}</div>
<div class=wrap>{_df_html(edges.round(2))}</div>
<h2>Queues and resources</h2><p class=muted>Average number waiting is time-weighted over the whole period. The longest queue grows with the length of the period
(the log covers {log.t.max() / 1440:.0f} days, each simulation run fewer), so it is shown but not scored.</p>
<div class=wrap>{_df_html(queue_table(val, spec).round(2))}</div>
<p class=muted>Simulated utilization: {", ".join(f"{k} {100 * v:.0f}%" for k, v in util.items())}.</p>
<h2>Validation (target: within {tol:.0%})</h2>
<p class=muted><span class=pill style="color:{STATUS['PASS'][0]};background:{STATUS['PASS'][1]}">pass</span> within {tol:.0%} (or under 0.1 min / 0.5 points)
· <span class=pill style="color:{STATUS['NOISE'][0]};background:{STATUS['NOISE'][1]}">noise</span> outside {tol:.0%} but within the spread between simulation runs
· <span class=pill style="color:{STATUS['FAIL'][0]};background:{STATUS['FAIL'][1]}">fail</span> a real gap.</p>
<div class=wrap>{status_table(val, tol)}</div>
{loop_html}
<h2>Distributions: log vs simulation</h2><p class=muted>Histograms scaled to the same area (probability density). Filled: log. Line: simulation, all runs pooled.</p>
<div class=grid>{"".join(figs)}</div>
</body></html>"""
    open(path, "w").write(doc)
