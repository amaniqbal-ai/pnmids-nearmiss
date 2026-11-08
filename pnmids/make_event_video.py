"""
v2: annotated near-miss videos with frame-by-frame identification.

For each event it writes
  <event_id>.mp4              the event window (3 s before to 2 s after), 1920x1340:
                              - the two road users boxed (red, blue), others grey
                              - a line between them coloured by live TTC (green > 3 s, amber, red < 1.5 s)
                              - each behaviour tagged on the frame it happens (BRAKING, ENTERED ON RED ...)
                              - a panel: the explanation, live TTC / gap / speeds, and a timeline
  <event_id>_storyboard.jpg   cause, response and closest-approach frames side by side

Usage:
  python -m pnmids.make_event_video --events results_v2/FI_events.csv \
      --traj-dir data/FI/derived_data/traj --video-dir FI/video --out videos_v2 \
      --event-ids FI_20250530_video-1_105 FI_20250121_video-1_5568
  (or --top 5 for the five highest-risk High events in the junction)
"""
import argparse
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config as C
from .explain import explain_event, load_video_tracks

RED, BLUE, GREY = (40, 40, 230), (230, 120, 40), (170, 170, 170)
TTC_COL = lambda t: (0, 0, 255) if t < 1.5 else ((0, 170, 255) if t < C.OPEN_TTC else (80, 180, 60))
TAG = {"braking": "BRAKING", "hard braking": "HARD BRAKING", "sharp acceleration": "SHARP ACCELERATION",
       "swerve": "SWERVE", "overtakes": "OVERTAKES", "entered on RED": "ENTERED ON RED",
       "entered on AMBER": "ENTERED ON AMBER"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
OUT_W, VID_H, PANEL_H = 1920, 1080, 260


def obb(img, r, color, thick, dx, dy):
    if any(pd.isna(v) for v in (r.cx, r.cy, r.w, r.h, r.r)):
        return
    pts = cv2.boxPoints(((r.cx - dx, r.cy - dy), (r.w, r.h), np.degrees(r.r))).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thick)


def label(img, text, x, y, color, scale=0.8):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x = int(min(max(x, 4), img.shape[1] - tw - 8)); y = int(min(max(y, th + 8), img.shape[0] - 6))
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 6), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def render_event(e, summary, tl, j, pix, video_path, out_dir, frame_offset):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    f0, f1 = int(j.index.min()), int(j.index.max())
    # crop window around the pair at closest approach (native pixels, no scaling when the video is >= 1080p)
    k = summary["closest_frame"]
    pa = pix[(pix.id == e.id_a) & (pix.frame == k)]; pb = pix[(pix.id == e.id_b) & (pix.frame == k)]
    cx = float(np.nanmean(list(pa.cx) + list(pb.cx))) if len(pa) + len(pb) else W0 / 2
    cy = float(np.nanmean(list(pa.cy) + list(pb.cy))) if len(pa) + len(pb) else H0 / 2
    cw, ch = min(OUT_W, W0), min(VID_H, H0)
    dx = int(np.clip(cx - cw / 2, 0, W0 - cw)); dy = int(np.clip(cy - ch / 2, 0, H0 - ch))
    seg = pix[(pix.frame >= f0) & (pix.frame <= f1)]
    by_frame = {f: g for f, g in seg.groupby("frame")}
    wrap = textwrap.wrap(summary["narrative"], 128)[:3]
    writer = cv2.VideoWriter(str(out_dir / f"{e.event_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, VID_H + PANEL_H))
    keyshots = {}
    marks = tl[tl.what.isin(list(TAG) + ["critical", "closest approach"])]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - frame_offset)
    for f in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        canvas = np.full((VID_H + PANEL_H, OUT_W, 3), 18, np.uint8)
        crop = img[dy:dy + ch, dx:dx + cw]
        if crop.shape[1] != OUT_W or crop.shape[0] != VID_H:
            crop = cv2.resize(crop, (OUT_W, VID_H))
            sx, sy = OUT_W / cw, VID_H / ch
        else:
            sx = sy = 1.0
        g = by_frame.get(f)
        centres = {}
        if g is not None:
            for r in g.itertuples():
                pair = r.id in (e.id_a, e.id_b)
                col = RED if r.id == e.id_a else (BLUE if r.id == e.id_b else GREY)
                obb(crop, r, col, 4 if pair else 1, dx, dy)
                if pair and not pd.isna(r.cx):
                    centres[r.id] = (int((r.cx - dx) * sx), int((r.cy - dy) * sy))
        row = j.loc[f] if f in j.index else None
        if row is not None and len(centres) == 2:
            t = row.ttc_s
            cv2.line(crop, centres[e.id_a], centres[e.id_b], TTC_COL(t), 3, cv2.LINE_AA)
            label(crop, f"red {e.type_a} {row.speed_a * 3.6:.0f} km/h", centres[e.id_a][0] + 30, centres[e.id_a][1] + 50, (200, 200, 255), 0.7)
            label(crop, f"blue {e.type_b} {row.speed_b * 3.6:.0f} km/h", centres[e.id_b][0] + 30, centres[e.id_b][1] - 40, (255, 210, 170), 0.7)
        # behaviour tags, shown for 1.5 s from the frame they happen
        for m in tl[(tl.frame <= f) & (tl.frame > f - 15) & tl.what.isin(list(TAG))].itertuples():
            rid = e.id_a if m.actor == "a" else e.id_b
            if rid in centres:
                label(crop, TAG[m.what], centres[rid][0] - 60, centres[rid][1] - 70, (0, 220, 255) if "RED" in TAG[m.what] else (255, 255, 255), 1.0)
        for m in tl[(tl.frame <= f) & (tl.frame > f - 15) & tl.what.isin(["critical", "closest approach"])].itertuples():
            label(crop, f"{m.what.upper()}: {m.detail}", 24, 60 if m.what == "critical" else 110, (0, 0, 255) if m.what == "critical" else (0, 220, 255), 1.0)
        label(crop, f"{e.event_id} | {e.severity} {e.conflict_type} | t = {f * C.DT:.1f} s", 24, VID_H - 24, (230, 230, 230), 0.8)
        canvas[:VID_H] = crop
        # panel: explanation, live values, timeline
        y = VID_H + 38
        for line in wrap:
            cv2.putText(canvas, line, (24, y), FONT, 0.8, (235, 235, 235), 2, cv2.LINE_AA); y += 36
        if row is not None:
            tt = "inf" if not np.isfinite(row.ttc_s) else f"{row.ttc_s:.1f} s"
            cv2.putText(canvas, f"TTC {tt}   gap {row.gap_m:.2f} m   red {row.speed_a * 3.6:.0f} km/h   blue {row.speed_b * 3.6:.0f} km/h",
                        (24, VID_H + 160), FONT, 0.8, TTC_COL(row.ttc_s), 2, cv2.LINE_AA)
        x0, x1, ty = 24, OUT_W - 24, VID_H + 215
        cv2.line(canvas, (x0, ty), (x1, ty), (90, 90, 90), 2)
        X = lambda fr: int(x0 + (fr - f0) / max(f1 - f0, 1) * (x1 - x0))
        for n, m in enumerate(marks.itertuples()):
            col = (0, 220, 255) if "RED" in str(m.what) or m.what == "closest approach" else ((0, 0, 255) if m.what == "critical" else (255, 255, 255))
            cv2.line(canvas, (X(m.frame), ty - 14), (X(m.frame), ty + 14), col, 3)
            txt = TAG.get(m.what, m.what.upper())
            (tw, _), _ = cv2.getTextSize(txt, FONT, 0.5, 1)
            tx = int(np.clip(X(m.frame) - tw / 2, 4, OUT_W - tw - 4))
            cv2.putText(canvas, txt, (tx, ty + 38 if n % 2 else ty - 20), FONT, 0.5, col, 1, cv2.LINE_AA)
        cv2.circle(canvas, (X(f), ty), 9, (255, 255, 255), -1)
        writer.write(canvas)
        for name, fr in (("cause", summary["cause_frame"]), ("response", summary["response_frame"]), ("closest", summary["closest_frame"])):
            if not pd.isna(fr) and int(fr) == f:
                keyshots[name] = canvas[:VID_H].copy()
    writer.release(); cap.release()
    if keyshots:
        tiles = [cv2.resize(keyshots[n], (960, 540)) for n in ("cause", "response", "closest") if n in keyshots]
        cv2.imwrite(str(out_dir / f"{e.event_id}_storyboard.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", default="videos_v2")
    ap.add_argument("--event-ids", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--frame-offset", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(a.events)
    if a.event_ids:
        sel = ev[ev.event_id.isin(a.event_ids)]
    else:
        sel = ev[(ev.status == "Accepted") & (ev.severity == "High") & (ev.zone == "junction")]
        sel = sel.sort_values("risk_score", ascending=False).head(a.top)
    rows = []
    for video, g in sel.groupby("video"):
        tr, route = load_video_tracks(a.traj_dir, video, set(g.id_a) | set(g.id_b))
        pix = pd.read_csv(Path(a.traj_dir) / f"{video}_Traj.csv", usecols=["frame", "id", "cx", "cy", "w", "h", "r"])
        for e in g.itertuples():
            summary, tl, j = explain_event(e, tr, route)
            if summary is None:
                continue
            render_event(e, summary, tl, j, pix, Path(a.video_dir) / f"{video}.mp4", out, a.frame_offset)
            tl.assign(event_id=e.event_id).to_csv(out / f"{e.event_id}_timeline.csv", index=False)
            rows.append(dict(event_id=e.event_id, **summary))
            print("wrote", e.event_id, "|", summary["narrative"])
    pd.DataFrame(rows).to_csv(out / "explained_events.csv", index=False)


if __name__ == "__main__":
    main()
