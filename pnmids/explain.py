"""
v2: near-miss IDENTIFICATION.

For each event it rebuilds the interaction frame by frame and answers:
  what happened, who did it, in which frame, and which action made the scene a near-miss.

Per frame (10 fps) for the pair:   gap between bodies, constant-velocity TTC, speeds
Per road user (behaviour events):  entered on red / amber, hard braking, sharp acceleration,
                                   swerve, overtake (passes the other)
Interaction milestones:            conflict starts (TTC < 3 s), critical (TTC < 1.5 s),
                                   minimum TTC, closest approach
Then a rule-based attribution gives CAUSE -> RESPONSE -> OUTCOME and one sentence.

Outputs (per run):
  explained_events.csv   one row per event: cause, response, outcome, narrative, key frames
  timeline.csv           one row per behaviour event / milestone: frame, time, actor, what, detail

Usage:
  python -m pnmids.explain --events results_v2/FI_events.csv --traj-dir data/FI/derived_data/traj \
      --top 20 --out results_v2/explained
  python -m pnmids.explain --events results_v2/FI_events.csv --traj-dir data/FI/derived_data/traj \
      --event-ids FI_20250530_video-1_105 FI_20250121_video-1_5568 --out results_v2/explained
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .engine import body_circles
from .kinematics import load_tracks

# behaviour thresholds (engineering defaults, to be tuned in WP3)
BRAKE = -1.0               # m/s2, braking = this deceleration held for BRAKE_RUN frames...
BRAKE_RUN = 10             # ...(1 s)...
BRAKE_DROP_KMH = 5.0       # ...and a speed drop of at least 5 km/h
HARD_BRAKE = -3.4          # m/s2 peak: labelled hard braking (same as the DRAC limit)
SHARP_ACCEL = 2.5          # m/s2, held for ACCEL_RUN frames
ACCEL_RUN = 5
MAX_PLAUSIBLE_ACCEL = 5.0  # m/s2, larger values are tracking noise and are ignored
MAX_PLAUSIBLE_DECEL = 9.0
MIN_RUN = 3                # frames (0.3 s) a behaviour must last to count
CAUSE_WINDOW_S = 6.0       # a violation counts as the cause only within this time before closest approach
SWERVE_YAW = 0.6           # rad/s while going straight
MIN_MOVING = 1.5           # m/s
CLOSE_HEADWAY = 1.0        # s, time headway counted as following too closely
PRE_S, POST_S = 3.0, 2.0   # window around the event


def _poly(x, y, h, L, W):
    from shapely.geometry import Polygon
    c, s = np.cos(h), np.sin(h)
    return Polygon([(x + c * dx - s * dy, y + s * dx + c * dy)
                    for dx, dy in [(L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)]])


def _dims(row, side):
    if row[f"type_{side}"] == "pedestrian":
        return 0.7, 0.7
    return row[f"length_med_{side}"], row[f"width_med_{side}"]


def pair_series(ev, tr):
    """Frame-by-frame interaction of the two road users around the event."""
    a = tr[tr.id == ev.id_a].set_index("frame"); b = tr[tr.id == ev.id_b].set_index("frame")
    f0 = int(ev.start_frame - PRE_S * C.FPS); f1 = int(ev.end_frame + POST_S * C.FPS)
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner").loc[f0:f1].copy()
    if j.empty:
        return j
    ts = np.arange(0, C.HORIZON_S + 1e-9, 0.1)
    offa, ra = body_circles(j.type_a.to_numpy(), j.length_med_a.to_numpy(float), j.width_med_a.to_numpy(float))
    offb, rb = body_circles(j.type_b.to_numpy(), j.length_med_b.to_numpy(float), j.width_med_b.to_numpy(float))
    gaps, ttcs = [], []
    for k, r in enumerate(j.itertuples()):
        La, Wa = _dims(j.iloc[k], "a"); Lb, Wb = _dims(j.iloc[k], "b")
        g = _poly(r.sx_a, r.sy_a, r.heading_a, La, Wa).distance(_poly(r.sx_b, r.sy_b, r.heading_b, Lb, Wb))
        gaps.append(g)
        # constant-velocity TTC with body circle chains
        pa = np.stack([r.sx_a + r.vx_a * ts, r.sy_a + r.vy_a * ts], -1)
        pb = np.stack([r.sx_b + r.vx_b * ts, r.sy_b + r.vy_b * ts], -1)
        ua = np.array([np.cos(r.heading_a), np.sin(r.heading_a)]); ub = np.array([np.cos(r.heading_b), np.sin(r.heading_b)])
        ca = pa[:, None, :] + offa[k][None, :, None] * ua[None, None, :]
        cb = pb[:, None, :] + offb[k][None, :, None] * ub[None, None, :]
        d = np.linalg.norm(ca[:, :, None, :] - cb[:, None, :, :], axis=-1)
        d = np.nanmin(d.reshape(len(ts), -1), axis=1)
        hit = np.where(d < ra[k] + rb[k])[0]
        closing = len(d) > 1 and d[1] < d[0]
        ttcs.append(ts[hit[0]] if len(hit) and (hit[0] > 0 or closing) else np.inf)
    j["gap_m"] = np.round(gaps, 2)
    j["ttc_s"] = ttcs
    j["time_s"] = j.index * C.DT
    ux, uy = np.cos(j.heading_a), np.sin(j.heading_a)
    j["b_ahead_m"] = (j.sx_b - j.sx_a) * ux + (j.sy_b - j.sy_a) * uy
    return j


def _onsets(mask, min_run=MIN_RUN):
    """Start index of every run of True that lasts at least min_run frames."""
    m = mask.to_numpy(bool); out = []; i = 0
    while i < len(m):
        if m[i]:
            k = i
            while k < len(m) and m[k]:
                k += 1
            if k - i >= min_run:
                out.append(i)
            i = k
        else:
            i += 1
    return out


def behaviour_events(j, ev, route):
    """Discrete things each road user did, with the frame they happened."""
    rows = []
    names = {"a": f"red {ev.type_a} #{ev.id_a}", "b": f"blue {ev.type_b} #{ev.id_b}"}
    moves = {"a": ev.move_a, "b": ev.move_b}
    for s in ("a", "b"):
        sp, acc, yaw = j[f"speed_{s}"], j[f"long_acc_{s}"], j[f"yaw_rate_{s}"]
        okd = acc >= -MAX_PLAUSIBLE_DECEL
        okp = acc <= MAX_PLAUSIBLE_ACCEL
        braking = (acc <= BRAKE) & okd & (sp.shift(5).fillna(sp) > MIN_MOVING)
        m = braking.to_numpy(bool); i = 0
        while i < len(m):
            if not m[i]:
                i += 1; continue
            k = i
            while k < len(m) and m[k]:
                k += 1
            v0 = sp.iloc[max(0, i - 5)] * 3.6; v1 = sp.iloc[k - 1] * 3.6
            if k - i >= BRAKE_RUN and v0 - v1 >= BRAKE_DROP_KMH:
                peak = acc.iloc[i:k][okd.iloc[i:k]].min()
                rows.append(dict(frame=j.index[i], actor=s, who=names[s],
                                 what="hard braking" if peak <= HARD_BRAKE else "braking",
                                 detail=f"{v0:.0f} to {v1:.0f} km/h in {(k - i) * C.DT:.1f} s, peak {peak:.1f} m/s2"))
            i = k
        for i in _onsets((acc >= SHARP_ACCEL) & okp & (sp > MIN_MOVING), ACCEL_RUN):
            k = j.index[i]
            rows.append(dict(frame=k, actor=s, who=names[s], what="sharp acceleration",
                             detail=f"+{acc.iloc[i:i + 10].max():.1f} m/s2"))
        if moves[s] == "s":
            for i in _onsets((yaw.abs() >= SWERVE_YAW) & (sp > 3)):
                k = j.index[i]
                rows.append(dict(frame=k, actor=s, who=names[s], what="swerve",
                                 detail=f"turn rate {np.degrees(yaw.iloc[i]):.0f} deg/s while going straight"))
        rid = ev.id_a if s == "a" else ev.id_b
        if route is not None and rid in route.index:
            st = route.at[rid, "in_state"]; tin = route.at[rid, "in_time"]
            if st in ("r", "y") and np.isfinite(tin):
                rows.append(dict(frame=int(round(tin * C.FPS)), actor=s, who=names[s],
                                 what="entered on RED" if st == "r" else "entered on AMBER",
                                 detail="crossed the stop line against the signal"))
    # overtake: the order along red's heading flips while heading the same way
    same = (np.abs(np.cos(j.heading_a - j.heading_b)) > np.cos(np.radians(37))) & (np.cos(j.heading_a - j.heading_b) > 0)
    sign = np.sign(j.b_ahead_m.where(j.b_ahead_m.abs() > 0.3))
    flips = [i for i in range(1, len(sign)) if same.iloc[i] and np.isfinite(sign.iloc[i]) and np.isfinite(sign.iloc[i - 1])
             and sign.iloc[i] != sign.iloc[i - 1]]
    if flips:
        i = flips[0]; passer = "b" if sign.iloc[i] > 0 else "a"
        other = "a" if passer == "b" else "b"
        rows.append(dict(frame=j.index[i], actor=passer, who=names[passer], what="overtakes",
                         detail=f"passes {names[other]} with {j.gap_m.iloc[max(0, i - 5):i + 5].min():.1f} m side gap"))
    return pd.DataFrame(rows)


def milestones(j):
    rows = []
    t = j.ttc_s
    def first(mask, what, det):
        idx = np.where(mask.to_numpy())[0]
        if len(idx):
            k = j.index[idx[0]]; rows.append(dict(frame=k, actor="pair", who="both", what=what, detail=det(k)))
    first(t < C.OPEN_TTC, "conflict starts", lambda k: f"TTC {j.at[k, 'ttc_s']:.1f} s, gap {j.at[k, 'gap_m']:.1f} m")
    first(t < 1.5, "critical", lambda k: f"TTC {j.at[k, 'ttc_s']:.1f} s, gap {j.at[k, 'gap_m']:.1f} m")
    if np.isfinite(t).any():
        k = t.idxmin(); rows.append(dict(frame=k, actor="pair", who="both", what="minimum TTC", detail=f"{t[k]:.2f} s"))
    k = j.gap_m.idxmin(); rows.append(dict(frame=k, actor="pair", who="both", what="closest approach",
                                           detail=f"{j.gap_m[k]:.2f} m at {max(j.at[k, 'speed_a'], j.at[k, 'speed_b']) * 3.6:.0f} km/h"))
    return pd.DataFrame(rows)


def attribute(ev, j, beh, ms):
    """CAUSE -> RESPONSE -> OUTCOME, rule based (explains, never recomputes the measures)."""
    names = {"a": f"red {ev.type_a}", "b": f"blue {ev.type_b}"}
    crit = ms.loc[ms.what == "critical", "frame"]
    start = ms.loc[ms.what == "conflict starts", "frame"]
    closest = ms.loc[ms.what == "closest approach", "frame"].iloc[0]
    t_crit = int(crit.iloc[0]) if len(crit) else int(closest)
    t_start = int(start.iloc[0]) if len(start) else int(j.index[0])
    cause, cause_frame = None, None
    context = ""
    win = CAUSE_WINDOW_S * C.FPS
    viol = beh[beh.what.isin(["entered on RED", "entered on AMBER"]) & (beh.frame <= closest)] if len(beh) else beh
    if len(viol):
        r = viol.sort_values("frame").iloc[0]
        if closest - r.frame <= win:
            cause, cause_frame = f"{names[r.actor]} {r.what.lower()}", r.frame
        else:
            context = f" Context: {names[r.actor]} had {r.what.lower()} {(closest - r.frame) * C.DT:.0f} s earlier."
    if cause is None and len(beh):
        pre = beh[beh.what.isin(["sharp acceleration", "swerve", "overtakes"]) &
                  (beh.frame >= t_start - 20) & (beh.frame <= closest)]
        if len(pre):
            r = pre.sort_values("frame").iloc[0]
            cause, cause_frame = f"{names[r.actor]} {r.what} ({r.detail})", r.frame
    if cause is None:
        ct = ev.conflict_type
        if ct == "rear end":
            fol = "a" if j.b_ahead_m.loc[t_start] > 0 else "b"
            v = j.at[t_start, f"speed_{fol}"]
            hw = j.at[t_start, "gap_m"] / v if v > 0.5 else np.inf
            cause = (f"{names[fol]} followed too closely (time headway {hw:.1f} s)" if hw < CLOSE_HEADWAY
                     else f"{names[fol]} closed in on a slower road user ahead")
        elif "l" in (ev.move_a, ev.move_b) or "t" in (ev.move_a, ev.move_b):
            s = "a" if ev.move_a in ("l", "t") else "b"; o = "b" if s == "a" else "a"
            cause = f"{names[s]} turned left across the path of {names[o]}"
        elif "r" in (ev.move_a, ev.move_b):
            s = "a" if ev.move_a == "r" else "b"; o = "b" if s == "a" else "a"
            cause = f"{names[s]} turned right into the path of {names[o]}"
        else:
            cause = "paths converged with no rule violation detected"
        cause_frame = t_start
    resp = beh[beh.what.isin(["hard braking", "braking", "swerve"]) & (beh.frame >= t_start - 10) & (beh.frame <= closest + 10)] if len(beh) else beh
    if len(resp):
        r = resp.sort_values("frame").iloc[0]
        lead = (closest - r.frame) * C.DT
        when = f"{lead:.1f} s before closest approach" if lead >= 0 else f"{-lead:.1f} s after closest approach"
        response = f"{names[r.actor]} {r.what} ({r.detail}), {when}"
        resp_frame = r.frame
    else:
        response, resp_frame = "no evasive action detected", np.nan
    mt = ms.loc[ms.what == "minimum TTC", "detail"]
    outcome = f"closest gap {j.gap_m.min():.2f} m" + (f", minimum TTC {mt.iloc[0]}" if len(mt) else "")
    narrative = (f"{ev.severity.upper()} {ev.conflict_type} near-miss: {cause}. "
                 f"Response: {response}. Outcome: {outcome}.{context}")
    return dict(cause=cause, cause_frame=cause_frame, response=response, response_frame=resp_frame,
                critical_frame=t_crit, closest_frame=int(closest), outcome=outcome, narrative=narrative)


def explain_event(ev, tracks, route):
    j = pair_series(ev, tracks)
    if j.empty:
        return None, None, None
    beh = behaviour_events(j, ev, route)
    ms = milestones(j)
    summary = attribute(ev, j, beh, ms)
    tl = pd.concat([beh, ms], ignore_index=True)
    tl["time_s"] = (tl.frame * C.DT).round(1)
    tl = tl.sort_values("frame").reset_index(drop=True)
    return summary, tl, j


def load_video_tracks(traj_dir, video, ids):
    raw = pd.read_csv(Path(traj_dir) / f"{video}_Traj.csv")
    tmp = Path("/tmp") / f"_explain_{video}.csv"
    raw[raw.id.isin(ids)].to_csv(tmp, index=False)
    tr = load_tracks(tmp); tmp.unlink()
    route_p = Path(traj_dir).parent / "route" / f"{video}_Route.csv"
    route = pd.read_csv(route_p).set_index("id") if route_p.exists() else None
    return tr, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out", default="results_v2/explained")
    ap.add_argument("--event-ids", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=20, help="otherwise: the N highest-risk accepted High events")
    ap.add_argument("--zone", default="junction")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(a.events)
    if a.event_ids:
        sel = ev[ev.event_id.isin(a.event_ids)]
    else:
        sel = ev[(ev.status == "Accepted") & (ev.severity == "High")]
        if a.zone:
            sel = sel[sel.zone == a.zone]
        sel = sel.sort_values("risk_score", ascending=False).head(a.top)
    sums, tls = [], []
    for video, g in sel.groupby("video"):
        tr, route = load_video_tracks(a.traj_dir, video, set(g.id_a) | set(g.id_b))
        for e in g.itertuples():
            s, tl, _ = explain_event(e, tr, route)
            if s is None:
                continue
            sums.append(dict(event_id=e.event_id, video=video, severity=e.severity, conflict_type=e.conflict_type,
                             red=f"{e.type_a} #{e.id_a}", blue=f"{e.type_b} #{e.id_b}", **s))
            tls.append(tl.assign(event_id=e.event_id))
            print(e.event_id, "->", s["narrative"])
    pd.DataFrame(sums).to_csv(out / "explained_events.csv", index=False)
    if tls:
        pd.concat(tls)[["event_id", "frame", "time_s", "who", "what", "detail"]].to_csv(out / "timeline.csv", index=False)
    print("wrote", out / "explained_events.csv", "and", out / "timeline.csv")


if __name__ == "__main__":
    main()
