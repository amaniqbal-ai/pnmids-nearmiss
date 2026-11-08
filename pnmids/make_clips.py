"""
WP3 ground truth, step 1: cut short review clips for a stratified sample of events.

For each sampled event it writes
  clips/<event_id>.mp4          6 s clip (3 s before / 3 s after the key frame),
                                road user A in red, B in blue, others in grey
  clips/<event_id>_preview.jpg  3 frames side by side (key-1 s, key, key+1 s)
and one annotation sheet  clips/annotation_sheet.csv  that you fill in.

Usage:
  python -m pnmids.make_clips --events results/FI_events.csv \
      --traj-dir data/FI/derived_data/traj --video-dir FI/video \
      --out clips/FI --n 60
Check the first clip: boxes must sit on the vehicles. If they are shifted by
one frame, rerun with --frame-offset 0 (default 1: traj frame 1 = first video frame).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

STRATA = {"High": 0.35, "Medium": 0.35, "Low": 0.15, "Rejected": 0.15}
PIX = ["frame", "id", "cx", "cy", "w", "h", "r"]


def sample_events(ev, n, seed):
    ev = ev.copy()
    ev["stratum"] = np.where(ev["status"] == "Rejected", "Rejected", ev["severity"])
    parts = []
    for s, share in STRATA.items():
        pool = ev[ev["stratum"] == s]
        k = min(len(pool), int(round(n * share)))
        if k:
            parts.append(pool.sample(k, random_state=seed))
    return pd.concat(parts).sort_values(["video", "key_frame"])


def draw_box(img, row, color, thick):
    if any(pd.isna(v) for v in (row.cx, row.cy, row.w, row.h, row.r)):
        return
    pts = cv2.boxPoints(((row.cx, row.cy), (row.w, row.h), np.degrees(row.r))).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", default="clips")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=3.0, help="seconds before and after the key frame")
    ap.add_argument("--frame-offset", type=int, default=1)
    ap.add_argument("--zone", default=None, help="only sample this zone, e.g. junction")
    ap.add_argument("--blind", action="store_true",
                    help="hide severity/TTC/PET on the clips and in the sheet (unbiased labelling)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(a.events)
    if a.zone:
        ev = ev[ev["zone"] == a.zone]
    sample = sample_events(ev, a.n, a.seed)
    print(f"sampled {len(sample)} events:", sample["stratum"].value_counts().to_dict())

    rows = []
    for video, group in sample.groupby("video"):
        vpath = Path(a.video_dir) / f"{video}.mp4"
        if not vpath.exists():
            print("missing video:", vpath); continue
        traj = pd.read_csv(Path(a.traj_dir) / f"{video}_Traj.csv", usecols=PIX)
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        half = int(round(a.seconds * fps))
        for e in group.itertuples():
            f0, f1 = max(1, e.key_frame - half), e.key_frame + half
            seg = traj[(traj["frame"] >= f0) & (traj["frame"] <= f1)]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - a.frame_offset)
            writer, previews = None, {}
            for f in range(f0, f1 + 1):
                ok, img = cap.read()
                if not ok:
                    break
                here = seg[seg["frame"] == f]
                for r in here.itertuples():
                    if r.id == e.id_a:
                        draw_box(img, r, (0, 0, 255), 3)
                    elif r.id == e.id_b:
                        draw_box(img, r, (255, 120, 0), 3)
                    else:
                        draw_box(img, r, (160, 160, 160), 1)
                if a.blind:
                    label = (f"{e.event_id} | {e.type_a}(red) vs {e.type_b}(blue) | t={f / 10:.1f}s"
                             + ("  <== KEY" if f == e.key_frame else ""))
                else:
                    label = (f"{e.event_id} | {e.stratum} | TTC {e.min_ttc:.2f}s PET {e.pet:.2f}s | "
                             f"{e.type_a}(red) vs {e.type_b}(blue) | t={f / 10:.1f}s"
                             + ("  <== KEY" if f == e.key_frame else ""))
                sc = max(0.8, img.shape[1] / 1600)          # readable on 4K video too
                bar = int(45 * sc)
                cv2.rectangle(img, (0, 0), (img.shape[1], bar), (0, 0, 0), -1)
                cv2.putText(img, label, (10, int(bar * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sc,
                            (255, 255, 255), max(2, int(2 * sc)))
                if writer is None:
                    writer = cv2.VideoWriter(str(out / f"{e.event_id}.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
                writer.write(img)
                if f in (e.key_frame - int(fps), e.key_frame, e.key_frame + int(fps)):
                    previews[f] = img.copy()
            if writer is not None:
                writer.release()
            if previews:
                tiles = [cv2.resize(previews[k], None, fx=0.5, fy=0.5) for k in sorted(previews)]
                cv2.imwrite(str(out / f"{e.event_id}_preview.jpg"), np.hstack(tiles))
            rows.append(dict(event_id=e.event_id, site=e.site, video=e.video, stratum=e.stratum,
                             zone=getattr(e, "zone", ""), engine_type=e.conflict_type,
                             min_ttc=e.min_ttc, pet=e.pet, clip=f"{e.event_id}.mp4",
                             is_real_conflict="", true_type="", tracking_ok="", notes=""))
        cap.release()
    sheet = pd.DataFrame(rows)
    if a.blind:
        sheet.to_csv(out / "sample_key.csv", index=False)          # engine info, do not open while labelling
        sheet = sheet.sample(frac=1, random_state=a.seed)            # shuffle so strata are not grouped
        sheet = sheet[["event_id", "clip", "is_real_conflict", "true_type", "tracking_ok", "notes"]]
    sheet.to_csv(out / "annotation_sheet.csv", index=False)
    print(f"wrote {len(sheet)} clips and {out / 'annotation_sheet.csv'}")
    print("Fill is_real_conflict with 1 (yes, a near-miss) or 0 (no). "
          "tracking_ok: 1 if boxes follow the right road users, else 0.")


if __name__ == "__main__":
    main()
