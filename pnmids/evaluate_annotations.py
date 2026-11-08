"""
WP3 ground truth, step 2: measure the engine against your manual labels.

Usage:
  python -m pnmids.evaluate_annotations --sheets clips/FI/annotation_sheet.csv \
      --events results/FI_events.csv --out results/annotation_eval

Outputs
  precision_by_stratum.csv   share of real conflicts in High / Medium / Low / Rejected
  estimated_overall.csv      precision of "Accepted High+Medium" weighted by stratum size
  threshold_sweep.csv/.png   how precision and recall move with the TTC / PET threshold
  critique_check.csv         were the rejected events really tracking errors?
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion (works for small samples)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--out", default="results/annotation_eval")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    lab = pd.concat([pd.read_csv(s) for s in a.sheets], ignore_index=True)
    lab = lab[lab["is_real_conflict"].astype(str).isin(["0", "1", "0.0", "1.0"])].copy()
    lab["y"] = lab["is_real_conflict"].astype(float).astype(int)
    ev = pd.concat([pd.read_csv(e) for e in a.events], ignore_index=True)
    ev["stratum"] = np.where(ev["status"] == "Rejected", "Rejected", ev["severity"])
    # blind sheets carry only event_id: bring the engine's values back in
    need = [c for c in ["stratum", "min_ttc", "pet", "conflict_type", "pair_kind", "zone"] if c not in lab.columns]
    if need:
        lab = lab.merge(ev[["event_id"] + need], on="event_id", how="left")
    print(f"{len(lab)} labelled events")

    rows = []
    for s, g in lab.groupby("stratum"):
        lo, hi = wilson(g["y"].sum(), len(g))
        rows.append(dict(stratum=s, labelled=len(g), real=int(g["y"].sum()),
                         precision=round(g["y"].mean(), 3), ci95_low=lo, ci95_high=hi,
                         events_in_stratum=int((ev["stratum"] == s).sum())))
    by = pd.DataFrame(rows)
    by.to_csv(out / "precision_by_stratum.csv", index=False)
    print(by.to_string(index=False))

    hm = by[by["stratum"].isin(["High", "Medium"])]
    if len(hm):
        w = hm["events_in_stratum"] / hm["events_in_stratum"].sum()
        est = float((w * hm["precision"]).sum())
        pd.DataFrame([dict(rule="Accepted High+Medium", weighted_precision=round(est, 3))]).to_csv(
            out / "estimated_overall.csv", index=False)
        print("estimated precision of Accepted High+Medium:", round(est, 3))

    # threshold sweep on the labelled sample (recall is relative to labelled positives)
    sweep = []
    pos = lab["y"].sum()
    for thr in np.round(np.arange(0.5, 3.01, 0.25), 2):
        for kind, col, ok in [("TTC", "min_ttc", True), ("PET", "pet", True)]:
            flag = lab[col] < thr
            tp = int((flag & (lab["y"] == 1)).sum()); fp = int((flag & (lab["y"] == 0)).sum())
            sweep.append(dict(measure=kind, threshold_s=thr, flagged=int(flag.sum()),
                              precision=round(tp / (tp + fp), 3) if tp + fp else np.nan,
                              recall=round(tp / pos, 3) if pos else np.nan))
    sw = pd.DataFrame(sweep)
    sw.to_csv(out / "threshold_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k in zip(axes, ["TTC", "PET"]):
        s = sw[sw["measure"] == k]
        ax.plot(s["threshold_s"], s["precision"], "o-", label="precision")
        ax.plot(s["threshold_s"], s["recall"], "s-", label="recall (labelled sample)")
        ax.set_title(f"{k} threshold sweep"); ax.set_xlabel(f"{k} threshold (s)"); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "threshold_sweep.png", dpi=150); plt.close(fig)

    if "tracking_ok" in lab.columns:
        rej = lab[lab["stratum"] == "Rejected"]
        t = rej["tracking_ok"].astype(str).isin(["0", "0.0"])
        pd.DataFrame([dict(rejected_labelled=len(rej), really_tracking_error=int(t.sum()),
                           were_real_conflicts=int(rej["y"].sum()))]).to_csv(out / "critique_check.csv", index=False)
    print("results in", out)


if __name__ == "__main__":
    main()
