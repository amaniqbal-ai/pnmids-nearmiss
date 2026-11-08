"""
Ablation: compare two pipeline runs side by side (e.g. v2 vs pilot-style).

Usage:
  python -m pnmids.compare_runs --runs results_v2 results_ablation \
      --names "v2 (shape + gates)" "pilot-style (1 circle, no gate)" --out results_v2/ablation.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MV = {"car", "truck", "bus", "tricycle", "van", "trailer"}


def summarise(run, sites):
    R = Path(run)
    ev = pd.concat([pd.read_csv(R / f"{s}_events.csv") for s in sites], ignore_index=True)
    m = pd.concat([pd.read_csv(R / f"{s}_dataset_match.csv") for s in sites], ignore_index=True)
    sm = pd.concat([pd.read_csv(R / f"{s}_summary.csv") for s in sites], ignore_index=True)
    hours = sm["minutes"].sum() / 60
    acc = ev[(ev["status"] == "Accepted") & ev["severity"].isin(["High", "Medium"])]
    mv = acc[acc["type_a"].isin(MV) & acc["type_b"].isin(MV)]
    mt = m[m["matched_any"]]
    row = {
        "hours": round(hours, 2),
        "High+Medium per hour (all users)": round(len(acc) / hours),
        "High per hour (all users)": round((acc["severity"] == "High").sum() / hours),
        "High+Medium per hour (motor vehicles)": round(len(mv) / hours),
        "dataset conflicts found (any event)": round(m["matched_any"].mean(), 3),
        "dataset conflicts found (accepted High/Medium)": round(m["our_event_id"].isin(set(acc["event_id"])).mean(), 3),
        "conflict-type agreement": round((mt["our_type"] == mt["conflict_type"]).mean(), 3),
        "TTC mean abs diff vs dataset (s)": round((mt["TTC"] - mt["our_min_ttc"]).abs().mean(), 3),
        "rejected by Critique": round((ev["status"] == "Rejected").mean(), 3),
        "vehicle-pedestrian High+Medium": int((acc["pair_kind"] == "vehicle-pedestrian").sum()),
    }
    if "zone" in acc.columns:
        for z in ["junction", "approach", "far"]:
            row[f"share in {z} zone"] = round((acc["zone"] == z).mean(), 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--sites", nargs="+", default=["FI", "FIDRT"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = a.names or a.runs
    table = pd.DataFrame({n: summarise(r, a.sites) for n, r in zip(names, a.runs)})
    print(table.to_string())
    if a.out:
        table.to_csv(a.out)


if __name__ == "__main__":
    main()
