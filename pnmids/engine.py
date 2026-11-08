"""
Step 2-5: the PNMIDS Layer 3 safety engine (v2).

For every frame it screens all pairs of road users, computes TTC, PET,
DRAC and Delta-V, opens/updates/closes conflict events, and classifies
each event (High / Medium / Low) with a 0-100 risk score.
It only reads tracked trajectories (never the dataset's own conflict file).

v2 changes over the pilot logic (see config.py):
  * real body shape: each road user is a chain of 1-4 circles built from
    its measured length and width (pilot: one fixed circle per class)
  * relative-speed gate: pairs closing slower than 0.3 m/s are ignored
  * PET grid placed on the junction box estimated from the data
"""
import numpy as np
import pandas as pd

from . import config as C


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def body_circles(types, length, width):
    """Circle chain for every row: offsets along the heading (N, K) and radius (N,)."""
    K = C.MAX_CIRCLES
    n = len(types)
    offs = np.full((n, K), np.nan)
    rad = np.zeros(n)
    ped = types == "pedestrian"
    if C.SHAPE_MODEL == "circle":
        rad[:] = [C.RADIUS.get(t, C.DEFAULT_RADIUS) for t in types]
        offs[:, 0] = 0.0
        return offs, rad
    W = np.clip(np.nan_to_num(width, nan=1.0), 0.4, 3.2)
    L = np.maximum(np.nan_to_num(length, nan=1.0), W)
    rad = W / 2
    cnt = np.clip(np.round(L / W), 1, K).astype(int)
    for k in range(1, K + 1):
        sel = cnt == k
        if not sel.any():
            continue
        if k == 1:
            offs[sel, 0] = 0.0
        else:
            half = (L[sel] / 2 - rad[sel])[:, None]
            offs[sel, :k] = half * np.linspace(-1, 1, k)[None, :]
    rad[ped] = C.PED_RADIUS
    offs[ped] = np.nan
    offs[ped, 0] = 0.0
    return offs, rad


class SafetyEngine:
    def __init__(self, tracks, axis_offset=0.0, box=None):
        self.df = tracks.sort_values(["frame", "id"]).reset_index(drop=True)
        self.axis_off = axis_offset
        self.ts = np.arange(0, C.HORIZON_S + 1e-9, C.SAMPLE_S)
        self.close_frames = int(round(C.CLOSE_IDLE_S * C.FPS))
        self.active, self.closed = {}, []
        # PET occupancy grid on the junction box (+ margin)
        if box is None:
            cx, cy = C.JUNCTION_CENTRE
            box = (cx - C.PET_GRID_HALF_M, cx + C.PET_GRID_HALF_M, cy - C.PET_GRID_HALF_M, cy + C.PET_GRID_HALF_M)
            margin = 0.0
        else:
            margin = C.PET_GRID_MARGIN_M
        self.gx0, self.gy0 = box[0] - margin, box[2] - margin
        self.GX = int(np.ceil((box[1] - box[0] + 2 * margin) / C.PET_CELL_M))
        self.GY = int(np.ceil((box[3] - box[2] + 2 * margin) / C.PET_CELL_M))
        self.cell_id = -np.ones((self.GX, self.GY), dtype=np.int64)
        self.cell_t = np.zeros((self.GX, self.GY))
        self.cell_h = np.zeros((self.GX, self.GY))
        self._offsets = {}

    # ------------------------------------------------------------------
    def _future(self, x, y, v, h, w):
        """Predicted centres (n, S, 2) and headings (n, S): constant velocity,
        or CTRV for turning users. A predicted turn ends when the heading
        reaches the next road axis; the user then continues straight."""
        ts = self.ts
        H = np.repeat(h[:, None], len(ts), axis=1)
        X = x[:, None] + (v * np.cos(h))[:, None] * ts[None, :]
        Y = y[:, None] + (v * np.sin(h))[:, None] * ts[None, :]
        turning = (np.abs(w) > C.YAW_RATE_TURNING) & (v > 1.0)
        for i in np.where(turning)[0]:
            wi, hi, vi = w[i], h[i], v[i]
            rel = np.mod(hi - self.axis_off, np.pi / 2)
            delta = (np.pi / 2 - rel) if wi > 0 else rel
            if delta < np.radians(3):       # already on an axis: the turn is ending
                continue
            te = min(delta / abs(wi), ts[-1])
            tt = np.minimum(ts, te)
            xa = x[i] + vi / wi * (np.sin(hi + wi * tt) - np.sin(hi))
            ya = y[i] - vi / wi * (np.cos(hi + wi * tt) - np.cos(hi))
            he = hi + wi * te
            rest = np.maximum(ts - te, 0)
            X[i] = xa + vi * np.cos(he) * rest
            Y[i] = ya + vi * np.sin(he) * rest
            H[i] = hi + wi * tt
        return np.stack([X, Y], axis=-1), H

    def _cells(self, r):
        r = round(float(r), 1)
        if r not in self._offsets:
            k = int(np.ceil(r / C.PET_CELL_M))
            off = [(dx, dy) for dx in range(-k, k + 1) for dy in range(-k, k + 1)
                   if np.hypot(dx, dy) * C.PET_CELL_M <= r + 1e-9]
            self._offsets[r] = np.array(off, dtype=int)
        return self._offsets[r]

    # ------------------------------------------------------------------
    def _update_event(self, key, f, **m):
        ev = self.active.get(key)
        if ev is None:
            ev = dict(id_a=key[0], id_b=key[1], start_frame=f, last_frame=f,
                      min_ttc=np.inf, frame_min_ttc=np.nan, ttc_frames=0,
                      speed_kmh_at_ttc=np.nan, relation=None, head_diff_deg=np.nan,
                      pet=np.nan, frame_pet=np.nan, speed_kmh_at_pet=np.nan,
                      pet_angle_deg=np.nan, drac_max=0.0, delta_v_kmh=0.0,
                      braking_a=0.0, braking_b=0.0)
            self.active[key] = ev
        ev["last_frame"] = f
        if "ttc" in m:
            ev["ttc_frames"] += 1
            ev["drac_max"] = max(ev["drac_max"], m["drac"])
            if m["ttc"] < ev["min_ttc"]:
                ev.update(min_ttc=m["ttc"], frame_min_ttc=f,
                          speed_kmh_at_ttc=m["speed_kmh"], relation=m["relation"],
                          head_diff_deg=m["hd_deg"], delta_v_kmh=m["dv_kmh"])
            ev["braking_a"] = min(ev["braking_a"], m["acc_a"])
            ev["braking_b"] = min(ev["braking_b"], m["acc_b"])
        if "pet" in m and (np.isnan(ev["pet"]) or m["pet"] < ev["pet"]):
            ev.update(pet=m["pet"], frame_pet=f, speed_kmh_at_pet=m["speed_kmh"],
                      pet_angle_deg=m["angle_deg"])
            if m.get("dv_kmh", 0) > ev["delta_v_kmh"] and np.isinf(ev["min_ttc"]):
                ev["delta_v_kmh"] = m["dv_kmh"]

    def _close_idle(self, f):
        for key in [k for k, e in self.active.items() if e["last_frame"] < f - self.close_frames]:
            self.closed.append(self.active.pop(key))

    # ------------------------------------------------------------------
    def _pet_step(self, t, ids, x, y, h, offs, r, use):
        """Each user occupies the cells under its body (circle chain)."""
        occ = {}
        for i in np.where(use)[0]:
            cells = set()
            for off in offs[i][~np.isnan(offs[i])]:
                px, py = x[i] + off * np.cos(h[i]), y[i] + off * np.sin(h[i])
                ci = int(np.floor((px - self.gx0) / C.PET_CELL_M))
                cj = int(np.floor((py - self.gy0) / C.PET_CELL_M))
                for dx, dy in self._cells(r[i]):
                    a, b = ci + dx, cj + dy
                    if 0 <= a < self.GX and 0 <= b < self.GY:
                        cells.add((a, b))
            for c in cells:
                occ.setdefault(c, []).append(i)
        out = []
        min_ang = np.radians(C.PET_MIN_ANGLE_DEG)
        for (a, b), lst in occ.items():
            here = {ids[i] for i in lst}
            prev = self.cell_id[a, b]
            for i in lst:
                if prev >= 0 and prev not in here:
                    ang = abs(_wrap(h[i] - self.cell_h[a, b]))
                    if ang > min_ang:
                        out.append((prev, ids[i], t - self.cell_t[a, b], i, np.degrees(ang)))
            for p in range(len(lst)):
                for q in range(p + 1, len(lst)):
                    i, j = lst[p], lst[q]
                    if ids[i] != ids[j]:
                        ang = abs(_wrap(h[i] - h[j]))
                        if ang > min_ang:
                            out.append((ids[i], ids[j], 0.0, j, np.degrees(ang)))
            newcomer = next((i for i in lst if ids[i] != prev), lst[0])
            self.cell_id[a, b] = ids[newcomer]
            self.cell_t[a, b] = t
            self.cell_h[a, b] = h[newcomer]
        return out

    # ------------------------------------------------------------------
    def run(self):
        df = self.df
        fr = df["frame"].to_numpy()
        splits = np.flatnonzero(np.diff(fr)) + 1
        typ = df["type"].to_numpy()
        L_all = df["length_med"].to_numpy(float)
        W_all = df["width_med"].to_numpy(float)
        offs_all, rad_all = body_circles(typ, L_all, W_all)
        mass_all = np.array([C.MASS.get(t, C.DEFAULT_MASS) for t in typ], float)
        cols = dict(
            id=df["id"].to_numpy(), x=df["sx"].to_numpy(), y=df["sy"].to_numpy(),
            vx=df["vx"].to_numpy(), vy=df["vy"].to_numpy(), v=df["speed"].to_numpy(),
            h=df["heading"].to_numpy(), w=df["yaw_rate"].to_numpy(), acc=df["long_acc"].to_numpy(),
            L=np.nan_to_num(L_all, nan=1.0), W=np.nan_to_num(W_all, nan=1.0))
        isped = typ == "pedestrian"
        rear_cos = np.cos(np.radians(C.REAR_END_MAX_ANGLE_DEG))

        for sl in np.split(np.arange(len(df)), splits):
            f = int(fr[sl[0]]); t = f * C.DT
            ids, x, y = cols["id"][sl], cols["x"][sl], cols["y"][sl]
            vx, vy, v, h, w = cols["vx"][sl], cols["vy"][sl], cols["v"][sl], cols["h"][sl], cols["w"][sl]
            L, Wd, acc = cols["L"][sl], cols["W"][sl], cols["acc"][sl]
            offs, r, m, ped = offs_all[sl], rad_all[sl], mass_all[sl], isped[sl]
            use = ~(ped & (v < C.STATIONARY_PED_SPEED))

            # ---- PET ----
            for a, b, pet, k, ang in self._pet_step(t, ids, x, y, h, offs, r, use):
                if pet < C.OPEN_PET:
                    ia, ib = np.where(ids == a)[0], np.where(ids == b)[0]
                    dv = 0.0
                    if len(ia) and len(ib):
                        i, j = ia[0], ib[0]
                        rel = np.hypot(vx[i] - vx[j], vy[i] - vy[j])
                        dv = max(m[j], m[i]) / (m[i] + m[j]) * rel * 3.6
                    self._update_event((min(a, b), max(a, b)), f, pet=pet,
                                       speed_kmh=v[k] * 3.6, angle_deg=ang, dv_kmh=dv)

            # ---- pair screening for TTC ----
            n = len(ids)
            if n >= 2:
                I, J = np.triu_indices(n, 1)
                d0 = np.hypot(x[I] - x[J], y[I] - y[J])
                ext = np.maximum(L, 2 * r) / 2
                relv = np.hypot(vx[I] - vx[J], vy[I] - vy[J])
                keep = (use[I] & use[J] & ~(ped[I] & ped[J])
                        & ~((v[I] < C.BOTH_SLOW_SPEED) & (v[J] < C.BOTH_SLOW_SPEED))
                        & (relv > C.MIN_CLOSING_SPEED)
                        & (d0 < relv * C.HORIZON_S + ext[I] + ext[J] + 2))
                I, J = I[keep], J[keep]
                if len(I):
                    ttc = np.full(len(I), np.inf)
                    drac = np.zeros(len(I))
                    relation = np.array(["cross"] * len(I), dtype=object)
                    hd = np.abs(_wrap(h[I] - h[J]))

                    # rear-end (vehicle pairs, similar heading, bodies overlap sideways)
                    cand = ~ped[I] & ~ped[J] & (np.cos(hd) > rear_cos)
                    for p in np.where(cand)[0]:
                        i, j = I[p], J[p]
                        ux, uy = np.cos(h[i]), np.sin(h[i])
                        lon = (x[j] - x[i]) * ux + (y[j] - y[i]) * uy
                        fol, lea = (i, j) if lon > 0 else (j, i)
                        ux, uy = np.cos(h[fol]), np.sin(h[fol])
                        rx, ry = x[lea] - x[fol], y[lea] - y[fol]
                        lon, lat = rx * ux + ry * uy, rx * uy - ry * ux
                        D = lon - (L[fol] + L[lea]) / 2
                        if abs(lat) > (Wd[fol] + Wd[lea]) / 2 or D <= 0:
                            continue
                        closing = (vx[fol] - vx[lea]) * ux + (vy[fol] - vy[lea]) * uy
                        relation[p] = "rear end"
                        if closing > C.MIN_CLOSING_SPEED:
                            ttc[p] = D / closing
                            drac[p] = min(closing ** 2 / (2 * D), C.DRAC_CAP)

                    # v2 lateral gate: near-parallel or near-opposite pairs that are
                    # side by side (adjacent lanes) only conflict if one drifts
                    # sideways towards the other (lane change / swerve)
                    par = (np.abs(np.cos(hd)) > rear_cos) & (relation == "cross")
                    for p in np.where(par)[0]:
                        i, j = I[p], J[p]
                        ux, uy = np.cos(h[i]), np.sin(h[i])
                        lat = (x[j] - x[i]) * uy - (y[j] - y[i]) * ux
                        lat_rate = (vx[j] - vx[i]) * uy - (vy[j] - vy[i]) * ux
                        side_by_side = abs(lat) > (Wd[i] + Wd[j]) / 2
                        drifting_in = lat * lat_rate < 0 and abs(lat_rate) > C.MIN_CLOSING_SPEED
                        if side_by_side and not drifting_in:
                            relation[p] = "parallel"          # no conflict

                    # crossing / turning pairs: circle chains + CV or CTRV prediction
                    cross = relation == "cross"
                    if cross.any():
                        Pc, Hc = self._future(x, y, v, h, w)
                        circ = (Pc[:, None, :, :] + offs[:, :, None, None] *
                                np.stack([np.cos(Hc), np.sin(Hc)], -1)[:, None, :, :])   # (n,K,S,2)
                        Ic, Jc = I[cross], J[cross]
                        diff = circ[Ic][:, :, None] - circ[Jc][:, None, :]            # (P,K,K,S,2)
                        dist = np.sqrt((diff ** 2).sum(-1))
                        dmin = np.nanmin(dist.reshape(len(Ic), -1, len(self.ts)), axis=1)  # (P,S)
                        R = (r[Ic] + r[Jc])[:, None]
                        hit = dmin < R
                        first = np.where(hit.any(1), hit.argmax(1), -1)
                        tc = np.where(first >= 0, self.ts[np.maximum(first, 0)], np.inf)
                        apart = (first == 0) & (dmin[:, 1] >= dmin[:, 0])
                        tc[apart] = np.inf
                        ttc[cross] = tc
                        vmax = np.maximum(v[Ic], v[Jc])
                        with np.errstate(divide="ignore", invalid="ignore"):
                            drac[cross] = np.where(np.isfinite(tc),
                                                   np.minimum(np.where(tc > 0, vmax / (2 * tc), C.DRAC_CAP), C.DRAC_CAP), 0.0)

                    for p in np.where(ttc < C.OPEN_TTC)[0]:
                        i, j = I[p], J[p]
                        dv = np.hypot(vx[i] - vx[j], vy[i] - vy[j])
                        dvi, dvj = m[j] / (m[i] + m[j]) * dv, m[i] / (m[i] + m[j]) * dv
                        a, b = (i, j) if ids[i] < ids[j] else (j, i)
                        self._update_event(
                            (ids[a], ids[b]), f, ttc=ttc[p], drac=drac[p],
                            speed_kmh=max(v[i], v[j]) * 3.6, relation=relation[p],
                            hd_deg=np.degrees(hd[p]), dv_kmh=max(dvi, dvj) * 3.6,
                            acc_a=acc[a], acc_b=acc[b])
            self._close_idle(f)

        self.closed.extend(self.active.values())
        self.active = {}
        return self._finalise(pd.DataFrame(self.closed))

    # ------------------------------------------------------------------
    def _finalise(self, ev):
        if ev.empty:
            return ev
        ev["end_frame"] = ev["last_frame"]
        ev["min_ttc"] = ev["min_ttc"].replace(np.inf, np.nan)

        def conflict_type(r):
            if r["relation"] == "rear end":
                return "rear end"
            hd = r["head_diff_deg"] if not np.isnan(r["head_diff_deg"]) else r["pet_angle_deg"]
            if np.isnan(hd):
                return "unknown"
            return "sideswipe" if hd < 30 else ("head on" if hd > 150 else "angle")
        ev["conflict_type"] = ev.apply(conflict_type, axis=1)

        def severity(r):
            ttc, pet = r["min_ttc"], r["pet"]
            s_ttc, s_pet = r["speed_kmh_at_ttc"], r["speed_kmh_at_pet"]
            if ((not np.isnan(ttc) and ttc < C.ttc_crit(s_ttc) and s_ttc >= 10)
                    or (not np.isnan(pet) and pet < 1.0 and s_pet >= 15)
                    or r["drac_max"] > C.DRAC_LIMIT):
                return "High"
            if (not np.isnan(ttc) and ttc < 2.0) or (not np.isnan(pet) and pet < 1.5):
                return "Medium"
            return "Low"
        ev["severity"] = ev.apply(severity, axis=1)

        prox = np.maximum.reduce([
            np.clip(1 - ev["min_ttc"].fillna(np.inf) / C.OPEN_TTC, 0, 1),
            np.clip(1 - ev["pet"].fillna(np.inf) / C.OPEN_PET, 0, 1),
            np.clip(ev["drac_max"] / 6.0, 0, 1)])
        sev = np.clip(ev["delta_v_kmh"] / 50.0, 0, 1)
        ev["risk_score"] = (100 * (0.65 * prox + 0.35 * sev)).round(1)
        ev["key_frame"] = ev["frame_min_ttc"].fillna(ev["frame_pet"]).astype(int)
        return ev
