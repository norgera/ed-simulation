"""Load an event log into the standard schema: case_id, activity, timestamp, [resource], plus time in minutes `t`."""
import pandas as pd

def load_log(path, case_col="case_id", activity_col="activity", time_col="timestamp", resource_col="resource",
             cases_path=None, class_col=None, cases_case_col=None):
    """Return (log, classes). `log` has columns case_id, activity, resource, ts, t (minutes since first event).
    `classes` is a Series case_id -> class label (or None). The class comes from `class_col`, found either in the
    event log itself or in the optional cases table."""
    df = pd.read_csv(path)
    missing = [c for c in (case_col, activity_col, time_col) if c not in df.columns]
    if missing:
        raise ValueError(f"column(s) {missing} not in {path}; columns are {list(df.columns)}. Use the column-mapping options.")
    out = pd.DataFrame({"case_id": df[case_col].astype(str), "activity": df[activity_col].astype(str),
                        "ts": pd.to_datetime(df[time_col])})
    out["resource"] = df[resource_col].astype("string") if resource_col in df.columns else pd.NA
    out["_i"] = range(len(out))
    out = out.sort_values(["case_id", "ts", "_i"], kind="mergesort").drop(columns="_i").reset_index(drop=True)
    t0 = out.ts.min()
    out["t"] = (out.ts - t0).dt.total_seconds() / 60.0
    classes = None
    if class_col:
        if class_col in df.columns:
            src = pd.DataFrame({"case_id": df[case_col].astype(str), "c": df[class_col]}).dropna().drop_duplicates("case_id")
        elif cases_path:
            cs = pd.read_csv(cases_path); cc = cases_case_col or case_col
            src = pd.DataFrame({"case_id": cs[cc].astype(str), "c": cs[class_col]}).dropna().drop_duplicates("case_id")
        else:
            raise ValueError(f"class column {class_col!r} not found in the log; pass --cases with the table that has it")
        classes = src.set_index("case_id")["c"].map(lambda v: str(int(v)) if isinstance(v, float) and v == int(v) else str(v))
    out.attrs["t0"] = t0
    return out, classes
