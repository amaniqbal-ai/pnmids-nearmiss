"""
Run the PNMIDS Layer 3 pipeline on real drone trajectories.

Usage (from the folder that contains the 'pnmids' package):
    python -m pnmids.run_pipeline --data data/FI/derived_data --site FI --out results
    python -m pnmids.run_pipeline --data data/FI/derived_data --site FI --video 20250530_video-1
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .kinematics import load_tracks, road_axis_offset, junction_box
from .engine import SafetyEngine
from .agents import critique, add_context, compare_with_dataset

KEEP = ["status", "reject_reason", "severity", "risk_score", "conflict_type", "pair_kind",
        "id_a", "type_a", "id_b", "type_b", "start_frame", "end_frame", "key_frame",
        "zone", "min_ttc", "pet", "drac_max", "delta_v_kmh", "speed_kmh_at_ttc", "speed_kmh_at_pet",
        "braking_a", "braking_b", "x_a", "y_a", "x_b", "y_b", "speed_kmh_a", "speed_kmh_b",
        "entry_a", "move_a", "entry_b", "move_b", "signal_now_a", "signal_now_b",
        "stopline_state_a", "stopline_state_b", "red_light_runner_involved", "ttc_frames"]


def run_video(data, site, video, conflicts):
    t0 = time.time()
    traj = data / "traj" / f"{video}_Traj.csv"
    sig_p, route_p = data / "signal" / f"{video}_signal.csv", data / "route" / f"{video}_Route.csv"
    tracks = load_tracks(traj)
    box = junction_box(tracks)
    events = SafetyEngine(tracks, road_axis_offset(tracks), box).run()
    events = critique(events, tracks)
    signal = pd.read_csv(sig_p) if sig_p.exists() else None
    route = pd.read_csv(route_p, usecols=["id", "in_state"]) if route_p.exists() else None
    events = add_context(events, tracks, signal, route, box)
    events.insert(0, "event_id", [f"{site}_{video}_{i}" for i in events.index])
    events.insert(0, "video", video)
    events.insert(0, "site", site)
    matched, summary = compare_with_dataset(events, conflicts, f"{video}_Traj")
    summary.update(site=site, video=video, minutes=round(tracks["frame"].max() / 10 / 60, 1),
                   road_users=int(tracks["id"].nunique()),
                   accepted=int((events["status"] == "Accepted").sum()),
                   rejected=int((events["status"] == "Rejected").sum()),
                   high=int(((events["status"] == "Accepted") & (events["severity"] == "High")).sum()),
                   medium=int(((events["status"] == "Accepted") & (events["severity"] == "Medium")).sum()),
                   junction_box=[round(b, 1) for b in box],
                   runtime_s=round(time.time() - t0, 1))
    print(json.dumps(summary))
    return events[["site", "video", "event_id"] + KEEP], matched, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to <site>/derived_data")
    ap.add_argument("--site", required=True)
    ap.add_argument("--video", default=None, help="one video name, e.g. 20250530_video-1")
    ap.add_argument("--out", default="results")
    ap.add_argument("--shape", choices=["multicircle", "circle"], default=None, help="override SHAPE_MODEL")
    ap.add_argument("--min-closing", type=float, default=None, help="override MIN_CLOSING_SPEED (m/s)")
    a = ap.parse_args()
    from . import config as C
    if a.shape: C.SHAPE_MODEL = a.shape
    if a.min_closing is not None: C.MIN_CLOSING_SPEED = a.min_closing
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conflicts = pd.read_csv(data / "conflict" / f"{a.site}_conflict.csv")
    videos = [a.video] if a.video else sorted(p.name[:-9] for p in (data / "traj").glob("*_Traj.csv"))
    ev_all, m_all, s_all = [], [], []
    for v in videos:
        e, m, s = run_video(data, a.site, v, conflicts)
        ev_all.append(e); m_all.append(m.assign(site=a.site)); s_all.append(s)
    pd.concat(ev_all).to_csv(out / f"{a.site}_events.csv", index=False)
    pd.concat(m_all).drop(columns=["key"]).to_csv(out / f"{a.site}_dataset_match.csv", index=False)
    pd.DataFrame(s_all).to_csv(out / f"{a.site}_summary.csv", index=False)


if __name__ == "__main__":
    main()
