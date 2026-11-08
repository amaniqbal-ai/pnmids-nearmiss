"""
WP3 analysis on the pipeline outputs:
  1) sensitivity of event counts to the classification rules
  2) agreement with the dataset's own conflict list (like-for-like)
  3) overview tables and two figures
Run after run_pipeline:  python -m pnmids.wp3_analysis --results results --sites FI FIDRT
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MV = {"car", "truck", "bus", "tricycle", "van", "trailer"}   # what the dataset list covers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sites", nargs="+", default=["FI", "FIDRT"])
    a = ap.parse_args()
    R = Path(a.results)
    ev = pd.concat([pd.read_csv(R / f"{s}_events.csv") for s in a.sites], ignore_index=True)
    m = pd.concat([pd.read_csv(R / f"{s}_dataset_match.csv") for s in a.sites], ignore_index=True)
    sm = pd.concat([pd.read_csv(R / f"{s}_summary.csv") for s in a.sites], ignore_index=True)
    hours = sm.groupby("site")["minutes"].sum() / 60

    acc = ev[ev["status"] == "Accepted"].copy()
    acc["mv_pair"] = acc["type_a"].isin(MV) & acc["type_b"].isin(MV)
    acc["in_dataset"] = acc["event_id"].isin(set(m["our_event_id"].dropna()))
    spd = acc[["speed_kmh_at_ttc", "speed_kmh_at_pet"]].max(axis=1)
    hm = acc["severity"].isin(["High", "Medium"])
    variants = {
        "A  Annex rules as-is (High+Medium)": hm,
        "B  A + speed >= 10 km/h for every event": hm & (spd >= 10),
        "C  TTC < 1.5 s or PET < 1.0 s (pipeline-spec rule)": (acc["min_ttc"] < 1.5) | (acc["pet"] < 1.0),
        "D  High only (Annex)": acc["severity"].eq("High"),
    }
    rows = []
    for name, mask in variants.items():
        for scope, smask in [("all road users", True), ("motor vehicles only", acc["mv_pair"])]:
            sel = acc[mask & smask]
            rows.append({
                "rule": name, "scope": scope, "events": len(sel),
                "events_per_hour": round(len(sel) / hours.sum()),
                "dataset_conflicts_recovered": round(m["our_event_id"].isin(set(sel["event_id"])).mean(), 3),
                "share_of_events_also_in_dataset": round(sel["in_dataset"].mean(), 3) if len(sel) else np.nan,
            })
    sens = pd.DataFrame(rows)
    sens.to_csv(R / "WP3_sensitivity.csv", index=False)
    print(sens.to_string(index=False))

    over = (acc[hm].groupby(["site", "pair_kind", "severity"]).size().unstack(fill_value=0))
    over["per_hour"] = (over.sum(axis=1) / over.index.get_level_values(0).map(hours)).round(1)
    over.to_csv(R / "WP3_overview_by_pair_kind.csv")
    print(over)

    mt = m[m["matched_any"]]
    agree = pd.crosstab(mt["conflict_type"], mt["our_type"])
    agree.to_csv(R / "WP3_conflict_type_agreement.csv")
    stats = {
        "hours_analysed": round(hours.sum(), 2),
        "road_users": int(sm["road_users"].sum()),
        "dataset_conflicts": len(m),
        "dataset_conflicts_matched_by_any_event": round(m["matched_any"].mean(), 3),
        "conflict_type_agreement": round((mt["our_type"] == mt["conflict_type"]).mean(), 3),
        "ttc_correlation_with_dataset": round(mt[["TTC", "our_min_ttc"]].corr().iloc[0, 1], 3),
        "ttc_mean_abs_diff_s": round((mt["TTC"] - mt["our_min_ttc"]).abs().mean(), 3),
        "critique_rejected_share": round((ev["status"] == "Rejected").mean(), 3),
        "high_with_red_light_runner": int(acc.loc[acc["severity"] == "High", "red_light_runner_involved"].sum()),
    }
    pd.Series(stats).to_csv(R / "WP3_key_numbers.csv", header=["value"])
    print(pd.Series(stats))

    # ---- Figure 1: hotspots of accepted High events
    fig, axes = plt.subplots(1, len(a.sites), figsize=(7 * len(a.sites), 6))
    axes = np.atleast_1d(axes)
    colors = {"vehicle-vehicle": "#1f77b4", "vehicle-moped": "#ff7f0e",
              "moped-moped": "#2ca02c", "vehicle-pedestrian": "#d62728"}
    for ax, site in zip(axes, a.sites):
        h = acc[(acc["site"] == site) & (acc["severity"] == "High")]
        for k, c in colors.items():
            s = h[h["pair_kind"] == k]
            ax.scatter((s["x_a"] + s["x_b"]) / 2, (s["y_a"] + s["y_b"]) / 2, s=6, c=c, alpha=0.5, label=f"{k} ({len(s)})")
        ax.set_title(f"{site}: accepted High events ({hours[site]:.1f} h)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, markerscale=3)
    fig.tight_layout(); fig.savefig(R / "fig_hotspots_high.png", dpi=150); plt.close(fig)

    # ---- Figure 2: sensitivity
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    s_all = sens[sens["scope"] == "all road users"]; s_mv = sens[sens["scope"] == "motor vehicles only"]
    xs = np.arange(len(s_all)); wd = 0.38
    ax1.bar(xs - wd / 2, s_all["events_per_hour"], wd, label="events/hour, all users", color="#9ecae1")
    ax1.bar(xs + wd / 2, s_mv["events_per_hour"], wd, label="events/hour, motor vehicles", color="#3182bd")
    ax1.set_ylabel("events per hour"); ax1.set_xticks(xs, [r.split()[0] for r in s_all["rule"]])
    ax2 = ax1.twinx()
    ax2.plot(xs, s_all["dataset_conflicts_recovered"] * 100, "o-", color="#d62728", label="dataset conflicts recovered (%)")
    ax2.set_ylabel("% of dataset conflicts recovered"); ax2.set_ylim(0, 100)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title("WP3 sensitivity: stricter rules cut events but also miss dataset conflicts")
    fig.tight_layout(); fig.savefig(R / "fig_sensitivity.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
