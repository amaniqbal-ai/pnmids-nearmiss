"""
Step 1: load a trajectory CSV and compute kinematics the same way the
pilot's Layer 2 does (smoothing, 0.5 s velocity, heading, yaw rate,
anomaly flag). No artificial noise is added: the data is already real.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config as C

USECOLS = ["frame", "id", "type", "isReal", "cx_m", "cy_m",
           "length_med", "width_med", "entry_direction",
           "exit_direction", "overall_direction"]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_tracks(path):
    df = pd.read_csv(path, usecols=USECOLS)
    df = df.sort_values(["id", "frame"]).reset_index(drop=True)
    df["time"] = df["frame"] * C.DT
    k = int(round(C.VEL_WINDOW_S * C.FPS))           # 5 frames at 10 fps
    a = C.EMA_ALPHA

    parts = []
    for _, g in df.groupby("id", sort=False):
        g = g.copy()
        x, y = g["cx_m"].to_numpy(float), g["cy_m"].to_numpy(float)
        # exponential smoothing (same filter as the pilot tracker)
        sx = lfilter([a], [1, -(1 - a)], x, zi=[(1 - a) * x[0]])[0]
        sy = lfilter([a], [1, -(1 - a)], y, zi=[(1 - a) * y[0]])[0]
        # velocity = displacement over the last 0.5 s
        vx, vy = np.zeros_like(sx), np.zeros_like(sy)
        if len(sx) > 1:
            lag = np.minimum(np.arange(len(sx)), k)
            lag[0] = 1
            idx0 = np.arange(len(sx)) - lag
            idx0[0] = 0
            vx = (sx - sx[idx0]) / (lag * C.DT)
            vy = (sy - sy[idx0]) / (lag * C.DT)
            vx[0], vy[0] = vx[1] if len(vx) > 1 else 0, vy[1] if len(vy) > 1 else 0
        g["sx"], g["sy"], g["vx"], g["vy"] = sx, sy, vx, vy
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    df["speed"] = np.hypot(df["vx"], df["vy"])
    head = np.arctan2(df["vy"], df["vx"])
    head = head.where(df["speed"] >= C.HEADING_HOLD_SPEED)
    df["heading"] = head.groupby(df["id"]).ffill().groupby(df["id"]).bfill().fillna(0.0)

    grp = df.groupby("id")
    prev_head = grp["heading"].shift(k)
    df["yaw_rate"] = (_wrap(df["heading"] - prev_head) / C.VEL_WINDOW_S).fillna(0.0)
    df["long_acc"] = ((df["speed"] - grp["speed"].shift(k)) / C.VEL_WINDOW_S).fillna(0.0)

    # anomaly: raw detection far from where the smoothed track expected it
    px = grp["sx"].shift(1) + grp["vx"].shift(1) * C.DT
    py = grp["sy"].shift(1) + grp["vy"].shift(1) * C.DT
    jump = np.hypot(df["cx_m"] - px, df["cy_m"] - py)
    df["anomaly"] = (jump > C.ANOMALY_JUMP_M).fillna(False)
    return df


def road_axis_offset(df):
    """Estimate how far the road grid is rotated from the x-axis (radians).
    Used by CTRV to end a turn when the heading reaches the next road axis."""
    m = df[df["speed"] > 3]
    if m.empty:
        return 0.0
    h = np.mod(m["heading"].to_numpy(), np.pi / 2)        # fold into 0..90 deg
    ang = np.angle(np.mean(np.exp(1j * 4 * h))) / 4      # circular mean with period 90 deg
    return float(ang)


def junction_box(df):
    """v2: estimate the junction box from the data itself.
    x-range = where straight N-S / S-N traffic drives (width of the N-S road),
    y-range = where straight W-E / E-W traffic drives (width of the W-E road).
    Returns (xmin, xmax, ymin, ymax) in metres."""
    m = df[df["speed"] > 3]
    ns = m[m["overall_direction"].isin(["N-S", "S-N"])]
    we = m[m["overall_direction"].isin(["W-E", "E-W"])]
    if len(ns) < 50 or len(we) < 50:
        return (-20.0, 20.0, -20.0, 20.0)
    x0, x1 = ns["cx_m"].quantile([0.05, 0.95])
    y0, y1 = we["cy_m"].quantile([0.05, 0.95])
    return (float(x0), float(x1), float(y0), float(y1))
