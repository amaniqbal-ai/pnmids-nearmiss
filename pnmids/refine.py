"""
v3 refinement, learned from manual labels (120 clips, 100 sure labels).

Adds to each Accepted High/Medium event:
  obs_min_clearance_m   smallest real gap between the two bodies (oriented boxes)
  obs_secs_below_2m     how long they were closer than 2 m
  obs_max_brake_ms2     hardest braking by either user in the event window
  obs_overtake          1 if the follower passed the leader (order flipped)
  refined_type          'overtake' replaces 'rear end' when obs_overtake = 1
  p_real                probability a human would call it a real near-miss,
                        from the engine risk score (logistic calibration on the labels)
  alert_level           Alert (High) / Review (Medium) / Log (Low) / Discard (Rejected)

Window: from 1 s before the event to 3 s after it, so these are post-hoc
(offline) measures, available about 3 s after an event.

Usage:
  python -m pnmids.refine --events results_v2/FI_events.csv \
      --traj-dir data/FI/derived_data/traj --out results_v2/FI_events_refined.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .kinematics import load_tracks

# logistic calibration of risk_score -> P(real near-miss), fitted on the manual labels
CAL_A, CAL_B = None, None   # filled from calibration.json if present
ALERT = {"High": "Alert", "Medium": "Review", "Low": "Log"}


def _boxes_clearance(xa, ya, ha, La, Wa, xb, yb, hb, Lb, Wb):
    from shapely.geometry import Polygon
    def poly(x, y, h, L, W):
        c, s = np.cos(h), np.sin(h)
        return Polygon([(x + c*dx - s*dy, y + s*dx + c*dy)
                        for dx, dy in [(L/2, W/2), (L/2, -W/2), (-L/2, -W/2), (-L/2, W/2)]])
    return poly(xa, ya, ha, La, Wa).distance(poly(xb, yb, hb, Lb, Wb))


def observed(ev, tr):
    a = tr[tr.id == ev.id_a].set_index("frame"); b = tr[tr.id == ev.id_b].set_index("frame")
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner").loc[ev.start_frame - 10: ev.end_frame + 30]
    if j.empty:
        return {}
    dims = lambda t, L, W: (0.7, 0.7) if t == "pedestrian" else (L, W)
    cl = []
    for r in j.itertuples():
        La, Wa = dims(r.type_a, r.length_med_a, r.width_med_a); Lb, Wb = dims(r.type_b, r.length_med_b, r.width_med_b)
        cl.append(_boxes_clearance(r.sx_a, r.sy_a, r.heading_a, La, Wa, r.sx_b, r.sy_b, r.heading_b, Lb, Wb))
    cl = np.array(cl)
    hd = np.abs((j.heading_a - j.heading_b + np.pi) % (2 * np.pi) - np.pi)
    lon = (j.sx_b - j.sx_a) * np.cos(j.heading_a) + (j.sy_b - j.sy_a) * np.sin(j.heading_a)
    same_dir = (hd < np.radians(37)).mean() > 0.6
    flip = int(same_dir and np.sign(lon.iloc[0]) != np.sign(lon.iloc[-1]) and lon.abs().iloc[[0, -1]].min() > 0.5)
    return dict(obs_min_clearance_m=round(float(cl.min()), 2),
                obs_secs_below_2m=round(float((cl < 2).sum()) / 10, 1),
                obs_max_brake_ms2=round(float(max(0, -j.long_acc_a.min(), -j.long_acc_b.min())), 2),
                obs_overtake=flip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default=str(Path(__file__).with_name("calibration.json")))
    ap.add_argument("--all", action="store_true", help="also compute observed features for Low events (slower)")
    a = ap.parse_args()
    import json
    cal = json.load(open(a.calibration))
    ev = pd.read_csv(a.events)
    ev["alert_level"] = np.where(ev.status == "Rejected", "Discard", ev.severity.map(ALERT))
    z = cal["a"] + cal["b"] * ev["risk_score"]
    ev["p_real"] = (1 / (1 + np.exp(-z))).round(3)
    ev.loc[ev.status == "Rejected", "p_real"] = 0.0
    todo = ev[(ev.status == "Accepted") & (ev.severity.isin(["High", "Medium"]) | a.all)]
    rows = []
    for video, g in todo.groupby("video"):
        ids = set(g.id_a) | set(g.id_b)
        raw = pd.read_csv(Path(a.traj_dir) / f"{video}_Traj.csv")
        tmp = Path("/tmp") / f"_refine_{video}.csv"; raw[raw.id.isin(ids)].to_csv(tmp, index=False)
        tr = load_tracks(tmp); tmp.unlink()
        for e in g.itertuples():
            rows.append(dict(event_id=e.event_id, **observed(e, tr)))
        print(video, len(g), "events refined", flush=True)
    ev = ev.merge(pd.DataFrame(rows), on="event_id", how="left")
    ev["refined_type"] = np.where((ev.conflict_type == "rear end") & (ev.obs_overtake == 1), "overtake", ev.conflict_type)
    ev.to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
