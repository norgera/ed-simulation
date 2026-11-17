"""Baseline KPI summary from ed_cases.csv + ed_event_log.csv (mean/median/p90 by CTAS and admitted vs not)."""
import sys, pandas as pd, numpy as np
d = sys.argv[1] if len(sys.argv)>1 else "."
cs = pd.read_csv(f"{d}/ed_cases.csv", parse_dates=["arrival_ts","departure_ts"])
ev = pd.read_csv(f"{d}/ed_event_log.csv", parse_dates=["timestamp"])
w = ev.pivot_table(index="case_id", columns="activity", values="timestamp", aggfunc="first")
cs = cs.set_index("case_id").join(w)
m = lambda a,b: (cs[b]-cs[a]).dt.total_seconds()/60
cs["door_to_triage_min"]=m("Arrival","Triage start")
cs["door_to_PIA_min"]=m("Arrival","Physician initial assessment start")
cs["boarding_min"]=m("Boarding start (admit request)","Inpatient bed assigned")
cs["LOS_min"]=(cs.departure_ts-cs.arrival_ts).dt.total_seconds()/60
cs["group"]=np.where(cs.outcome=="LWBS","LWBS",np.where(cs.admitted,"Admitted","Non-admitted"))
def s(x): x=x.dropna(); return pd.Series(dict(n=len(x),mean=x.mean(),median=x.median(),p90=x.quantile(.9)))
rows=[]
for k in ["door_to_triage_min","door_to_PIA_min","boarding_min","LOS_min"]:
    for lab,g in [("All",cs),*[(f"CTAS {c}",cs[cs.ctas==c]) for c in range(1,6)],*[(x,cs[cs.group==x]) for x in ["Admitted","Non-admitted"]]]:
        r=s(g[k]); r["kpi"]=k; r["segment"]=lab; rows.append(r)
out=pd.DataFrame(rows)[["kpi","segment","n","mean","median","p90"]].round(1)
out.to_csv(f"{d}/baseline_kpis.csv",index=False)
pd.set_option("display.width",200); print(out.to_string(index=False))
print("\nLWBS rate: %.1f%%  | admit rate (of seen): %.1f%% | patients/day: %.0f" % (
  100*(cs.outcome=="LWBS").mean(), 100*cs[cs.outcome!="LWBS"].admitted.mean(), len(cs)/((cs.arrival_ts.max()-cs.arrival_ts.min()).days+1)))
# ED bed utilisation: bed assigned -> departure, sampled every 15 min
b=cs.dropna(subset=["ED bed assigned","departure_ts"])
t=pd.date_range(cs.arrival_ts.min().ceil("D")+pd.Timedelta(days=3),cs.arrival_ts.max()-pd.Timedelta(days=1),freq="15min")
st=b["ED bed assigned"].values; en=b["departure_ts"].values
occ=np.array([((st<=x)&(en>x)).sum() for x in t.values[::8]])
print("ED bed occupancy (of 30): mean %.1f (%.0f%%), p90 %.0f, share of time >=30: %.0f%%"%(occ.mean(),100*occ.mean()/30,np.quantile(occ,.9),100*(occ>=30).mean()))
bd=cs[cs.group=="Admitted"]; hrs=bd.boarding_min.sum()/60
print("Boarding: %.0f%% of admitted wait >60min; share of ED bed-hours used by boarders: %.0f%%"%(100*(bd.boarding_min>60).mean(),100*hrs/((b.departure_ts-b["ED bed assigned"]).dt.total_seconds().sum()/3600)))
