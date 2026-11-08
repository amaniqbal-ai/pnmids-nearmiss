"""
PNMIDS Layer 3 configuration.

Every value here comes from the PNMIDS Technical Annex unless marked
"ADAPTED" (changed because the real drone data differs from the pilot)
or "ASSUMED" (not in the Annex; engineering-judgement value that WP3
should test).
"""

# ---------------- Data ----------------
FPS = 10                      # ADAPTED: drone data is 10 fps (pilot ran at 20 fps)
DT = 1.0 / FPS

# Layer-2-equivalent kinematics (Annex: 0.5 s window, EMA weight 0.5 at 20 fps)
VEL_WINDOW_S = 0.5
EMA_WEIGHT_20FPS = 0.5
EMA_ALPHA = 1 - (1 - EMA_WEIGHT_20FPS) ** (20 / FPS)   # ADAPTED: same time-decay at 10 fps -> 0.75
ANOMALY_JUMP_M = 2.0          # Annex: >2 m from expected position = implausible
HEADING_HOLD_SPEED = 0.5      # ASSUMED: below this speed (m/s) keep the last heading

# ---------------- Road users ----------------
# Circle radius (m) for crossing TTC. Car 1.25 and pedestrian 0.35 are from the Annex.
RADIUS = {                    # others ASSUMED
    "car": 1.25, "pedestrian": 0.35, "moped": 0.5,
    "tricycle": 0.9, "truck": 1.8, "bus": 1.8, "van": 1.4, "trailer": 2.0,
}
# Mass (kg) for Delta-V. Car 1500 and pedestrian 75 are from the Annex.
MASS = {                      # others ASSUMED (moped includes rider)
    "car": 1500, "pedestrian": 75, "moped": 150,
    "tricycle": 400, "truck": 10000, "bus": 12000, "van": 2500, "trailer": 15000,
}
DEFAULT_RADIUS, DEFAULT_MASS = 1.25, 1500

# ---------------- Screening ----------------
STATIONARY_PED_SPEED = 0.3    # ASSUMED: standing pedestrians are excluded (Annex rule)
BOTH_SLOW_SPEED = 0.5         # ASSUMED: pairs where both are near-stationary are skipped
HORIZON_S = 3.0               # Annex: CTRV sampled up to 3 s
SAMPLE_S = 0.05               # Annex: sampled every 0.05 s
YAW_RATE_TURNING = 0.08       # Annex: rad/s, above this use CTRV
REAR_END_MAX_ANGLE_DEG = 37   # Annex: same lane, headings within ~37 deg

# ---------------- PET grid ----------------
PET_GRID_HALF_M = 20          # Annex: central 40 m x 40 m
PET_CELL_M = 1.0              # Annex: 1 m cells
PET_MIN_ANGLE_DEG = 30        # Annex: crossing angle above 30 deg
JUNCTION_CENTRE = (0.0, 0.0)  # fallback only; v2 estimates the junction box from each video
PET_GRID_MARGIN_M = 5         # v2: PET grid = junction box + 5 m on each side

# ---------------- Events ----------------
OPEN_TTC = 3.0                # Annex
OPEN_PET = 2.5                # Annex
CLOSE_IDLE_S = 1.0            # Annex
DRAC_LIMIT = 3.4              # Annex, m/s^2

# ---------------- Critique Agent ----------------
CRITIQUE_PRE_S = 0.6          # Annex: anomaly window starts 0.6 s before the event
CRITIQUE_MIN_TTC_S = 0.15     # Annex: TTC signal shorter than this with no PET -> reject


# ---------------- v2 improvements ----------------
SHAPE_MODEL = "multicircle"   # v2: "multicircle" (real length/width) or "circle" (pilot)
MAX_CIRCLES = 4               # v2: up to 4 circles along the body (bus ~4, car 2, moped 2)
PED_RADIUS = 0.35             # Annex pedestrian radius
MIN_CLOSING_SPEED = 0.3       # v2 (also in the pilot's JS code): ignore pairs closing slower than this, m/s
DRAC_CAP = 9.9                # pilot's JS caps DRAC at 9.9 m/s^2
APPROACH_BAND_M = 30          # v2: "approach" zone = within 30 m of the junction box


def ttc_crit(speed_kmh):
    """Annex speed-adjusted critical TTC (Swedish TCT principle)."""
    return min(2.0, max(1.0, 0.6 + 0.018 * speed_kmh))
