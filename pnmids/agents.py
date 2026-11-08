"""
Step 6-8: Critique Agent, Reporting context, and WP3 comparison.
"""
import numpy as np
import pandas as pd

from . import config as C

COMPASS = ["N", "E", "S", "W"]
TRAVEL = {"N": "S", "S": "N", "E": "W", "W": "E"}   # entering from N means travelling S


# ---------------------------------------------------------------- Critique
def critique(events, tracks):
    """Annex rules: reject if either track had an anomaly flag from 0.6 s
    before the event until it closed, or if the TTC signal lasted under
    0.15 s with no PET to support it."""
    anom = tracks.loc[tracks["anomaly"], ["id", "frame"]]
    anom_by_id = {k: np.sort(g["frame"].to_numpy()) for k, g in anom.groupby("id")}
    pre = int(round(C.CRITIQUE_PRE_S * C.FPS))

    def has_anom(i, f0, f1):
        a = anom_by_id.get(i)
        if a is None:
            return False
        k = np.searchsorted(a, f0)
        return k < len(a) and a[k] <= f1

    status, reason = [], []
    for r in events.itertuples():
        f0, f1 = r.start_frame - pre, r.end_frame
        if has_anom(r.id_a, f0, f1) or has_anom(r.id_b, f0, f1):
            status.append("Rejected"); reason.append("tracking anomaly (position jump > 2 m)")
        elif np.isnan(r.pet) and r.ttc_frames * C.DT < C.CRITIQUE_MIN_TTC_S:
            status.append("Rejected"); reason.append("TTC signal < 0.15 s with no PET support")
        else:
            status.append("Accepted"); reason.append("")
    events = events.copy()
    events["status"], events["reject_reason"] = status, reason
    return events


# ---------------------------------------------------------------- Context
def movement(entry, exit_):
    """s / l / r / t (U-turn) from entry and exit arms, right-hand traffic."""
    if entry not in TRAVEL or exit_ not in TRAVEL:
        return "unknown"
    if entry == exit_:
        return "t"
    heading_in = COMPASS.index(TRAVEL[entry])
    heading_out = COMPASS.index(exit_)
    diff = (heading_out - heading_in) % 4
    return {0: "s", 1: "r", 3: "l"}.get(diff, "unknown")


def add_context(events, tracks, signal, route, box=None):
    """What the Reporting Agent needs: who, where, which movement, which signal."""
    if events.empty:
        return events
    look = tracks.set_index(["id", "frame"])[["type", "cx_m", "cy_m", "speed",
                                               "entry_direction", "exit_direction"]]
    route_state = route.set_index("id")["in_state"].to_dict() if route is not None else {}
    out = events.copy()
    for side in ("a", "b"):
        keys = list(zip(out[f"id_{side}"], out["key_frame"]))
        sub = look.reindex(keys)
        # a user can be gone by the PET moment: fall back to its first/last row
        miss = sub["type"].isna().to_numpy()
        if miss.any():
            near = tracks.groupby("id").agg(type=("type", "first"), entry_direction=("entry_direction", "first"),
                                             exit_direction=("exit_direction", "first"))
            ids_m = np.array([k[0] for k in keys])[miss]
            sub.loc[miss, ["type", "entry_direction", "exit_direction"]] = near.reindex(ids_m).to_numpy()
        out[f"type_{side}"] = sub["type"].to_numpy()
        out[f"x_{side}"] = sub["cx_m"].round(2).to_numpy()
        out[f"y_{side}"] = sub["cy_m"].round(2).to_numpy()
        out[f"speed_kmh_{side}"] = (sub["speed"] * 3.6).round(1).to_numpy()
        out[f"entry_{side}"] = sub["entry_direction"].to_numpy()
        out[f"move_{side}"] = [movement(e, x) for e, x in zip(sub["entry_direction"], sub["exit_direction"])]
        out[f"stopline_state_{side}"] = [route_state.get(i, "") for i in out[f"id_{side}"]]
        out[f"signal_now_{side}"] = [
            signal_state(signal, e, mv, f * C.DT, ty)
            for e, mv, f, ty in zip(out[f"entry_{side}"], out[f"move_{side}"], out["key_frame"], out[f"type_{side}"])]
    out["pair_kind"] = [kind(a, b) for a, b in zip(out["type_a"], out["type_b"])]
    if box is not None:
        mx, my = (out["x_a"] + out["x_b"]) / 2, (out["y_a"] + out["y_b"]) / 2
        dx = np.maximum(np.maximum(box[0] - mx, mx - box[1]), 0)
        dy = np.maximum(np.maximum(box[2] - my, my - box[3]), 0)
        dist = np.hypot(dx, dy)
        out["zone"] = np.where(dist == 0, "junction", np.where(dist <= C.APPROACH_BAND_M, "approach", "far"))
    else:
        out["zone"] = ""
    out["red_light_runner_involved"] = (out["stopline_state_a"] == "r") | (out["stopline_state_b"] == "r")
    return out


def kind(a, b):
    s = {a, b}
    if "pedestrian" in s:
        return "vehicle-pedestrian"
    if "moped" in s:
        return "vehicle-moped" if s != {"moped"} else "moped-moped"
    return "vehicle-vehicle"


def signal_state(signal, entry, move, t, typ):
    if signal is None or typ == "pedestrian" or entry not in TRAVEL:
        return ""
    if move == "r":
        return "free right"                      # right turns have no signal group here
    grp = "l" if move in ("l", "t") else "s"
    s = signal[(signal["direction"] == entry) & (signal["turn"] == grp)
               & (signal["begin_time"] <= t) & (signal["end_time"] > t)]
    return s["state"].iloc[0] if len(s) else ""


# ---------------------------------------------------------------- WP3 comparison
def compare_with_dataset(events, ds, scene, tol_frames=10):
    """Match our events to the dataset's own conflict list (same pair,
    dataset frame inside our event window +/- 1 s). The dataset list is
    another algorithm's output, NOT human ground truth."""
    d = ds[ds["Scene"] == scene].copy()
    if "event_id" not in events.columns:
        events = events.assign(event_id=events.index.astype(str))
    d["key"] = [tuple(sorted((a, b))) for a, b in zip(d["Car 1 ID"], d["Car 2 ID"])]
    ev = events.copy()
    ev["key"] = list(zip(ev["id_a"], ev["id_b"]))
    by_key = {k: g for k, g in ev.groupby("key")}

    match_idx, our_ttc = [], []
    for r in d.itertuples():
        g = by_key.get(r.key)
        hit = None
        if g is not None:
            g = g[(g["start_frame"] - tol_frames <= r.Frame) & (g["end_frame"] + tol_frames >= r.Frame)]
            if len(g):
                hit = g.sort_values("risk_score", ascending=False).index[0]
        match_idx.append(hit)
    d["our_event"] = match_idx
    d["matched_any"] = d["our_event"].notna()
    d["our_event_id"] = [ev.at[i, "event_id"] if i is not None and not pd.isna(i) else "" for i in d["our_event"]]
    st = ev["status"].to_dict(); sv = ev["severity"].to_dict()
    d["our_status"] = [st.get(i, "") for i in d["our_event"]]
    d["our_severity"] = [sv.get(i, "") for i in d["our_event"]]
    d["our_min_ttc"] = [ev.at[i, "min_ttc"] if i is not None and not pd.isna(i) else np.nan for i in d["our_event"]]
    d["our_type"] = [ev.at[i, "conflict_type"] if i is not None and not pd.isna(i) else "" for i in d["our_event"]]

    acc_hm = ev[(ev["status"] == "Accepted") & ev["severity"].isin(["High", "Medium"])]
    ds_keys = set(zip(d["key"], d["Frame"]))
    def in_ds(r):
        return any(k == r.key and r.start_frame - tol_frames <= fr <= r.end_frame + tol_frames for k, fr in ds_keys)
    acc_hm_matched = sum(in_ds(r) for r in acc_hm.itertuples())

    summary = dict(
        scene=scene,
        dataset_conflicts=len(d),
        dataset_found_by_engine=int(d["matched_any"].sum()),
        dataset_found_as_accepted_high_or_medium=int(((d["our_status"] == "Accepted")
                                                      & d["our_severity"].isin(["High", "Medium"])).sum()),
        engine_accepted_high_medium=len(acc_hm),
        engine_events_also_in_dataset=acc_hm_matched,
        same_conflict_type_share=round(float((d.loc[d["matched_any"], "our_type"]
                                              == d.loc[d["matched_any"], "conflict_type"]).mean()), 3)
        if d["matched_any"].any() else np.nan,
    )
    m = d[d["matched_any"] & d["our_min_ttc"].notna()]
    summary["ttc_mean_abs_diff_s"] = round(float((m["our_min_ttc"] - m["TTC"]).abs().mean()), 3) if len(m) else np.nan
    return d, summary
