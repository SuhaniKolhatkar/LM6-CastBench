# -*- coding: utf-8 -*-
"""
flange_lhs_sampling.py  (v3 - fixes closed/capped-hub bug)

Generates a Design of Experiments (DOE) for the parametric flange using
Latin Hypercube Sampling (LHS).

BUG FIXED IN THIS VERSION:
Pad001 (the tapered boss) is an ANNULUS (outer circle boss_od, inner
circle = bore Db). PartDesign's TaperAngle shrinks BOTH boundaries of
that ring as it extrudes upward - not just the outer edge. For samples
with large Hb + large |taper| + modest Db, the INNER radius
(Db/2 - Hb*tan(|taper|)) was hitting zero before reaching the top of
the boss, pinching the through-hole shut inside the hub. That's the
"capped hub, no hole" defect you saw.

Fix: after computing Db/boss_od/Hb, we now check the inner radius at
the TOP of the boss and clamp |taper| (reduce it) whenever it would
close the hole below a minimum safe radius. This guarantees the bore
stays open all the way through the hub for every sample.

Output: doe.csv
"""

import numpy as np
import csv
import os
import math

# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------
N_SAMPLES = 10
SEED = 42
OUT_CSV = "doe.csv"

# Minimum geometric margins (mm) - protect topology / avoid closed holes
MIN_WALL_T = 6.0            # min (boss_od - Db)/2 -> boss annulus wall
MIN_RIM_MARGIN = 15.0       # min (D - boss_od)/2  -> flange rim outside boss
MIN_HOLE_CLEARANCE = 3.0    # min gap: bolt hole edge <-> boss_od / D
MIN_BORE_TOP_RADIUS = 5.0   # min inner (bore) radius remaining at TOP of boss
                             # (this is what stops the hub hole from closing)

# Independent variable bounds
BOUNDS = {
    "D":          (80.0, 140.0),
    "T":          (10.0, 25.0),
    "bore_ratio": (0.42, 0.52),   # Db = bore_ratio * D  (orig: 0.50)
    "wall_t":     (6.0, 10.0),    # boss_od = Db + 2*wall_t
    "Hb":         (8.0, 20.0),
    "taper":      (-15.0, 0.0),   # will be clamped per-sample if needed
    "dh":         (6.0, 10.0),
}

VAR_ORDER = ["D", "T", "bore_ratio", "wall_t", "Hb", "taper", "dh"]


# -----------------------------------------------------------------
# LHS
# -----------------------------------------------------------------
def lhs(n_samples, n_vars, seed=None):
    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, n_vars))
    for j in range(n_vars):
        cut_points = np.linspace(0, 1, n_samples + 1)
        low = cut_points[:n_samples]
        high = cut_points[1:n_samples + 1]
        points = low + rng.random(n_samples) * (high - low)
        rng.shuffle(points)
        result[:, j] = points
    return result


def scale_to_bounds(unit_samples, var_order, bounds):
    scaled = np.zeros_like(unit_samples)
    for j, name in enumerate(var_order):
        lo, hi = bounds[name]
        scaled[:, j] = lo + unit_samples[:, j] * (hi - lo)
    return scaled


# -----------------------------------------------------------------
# Dependent variable calculation + topology / open-hole safety
# -----------------------------------------------------------------
def clip(val, lo, hi):
    return max(lo, min(hi, val))


def compute_dependent(row):
    D, T, bore_ratio, wall_t, Hb, taper, dh = row
    clamped_flags = []

    # --- bore diameter, derived as a ratio of D ---
    Db = bore_ratio * D

    # --- boss outer diameter, derived from bore + wall thickness ---
    wall_t = max(wall_t, MIN_WALL_T)
    boss_od = Db + 2.0 * wall_t

    # --- enforce minimum rim margin between boss_od and D ---
    max_allowed_boss_od = D - 2.0 * MIN_RIM_MARGIN
    if boss_od > max_allowed_boss_od:
        boss_od = max_allowed_boss_od
        clamped_flags.append("boss_od_rim")

    if boss_od <= Db + 2.0 * MIN_WALL_T:
        boss_od = Db + 2.0 * MIN_WALL_T
        clamped_flags.append("boss_od_wall")

    # --- NEW: prevent taper from closing the bore inside the boss ---
    # Both the outer AND inner boundary of the tapered annulus shrink by
    # Hb * tan(|taper|) as the pad extrudes up. Clamp |taper| so the
    # inner (bore) radius at the TOP of the boss never drops below
    # MIN_BORE_TOP_RADIUS.
    taper_mag = abs(taper)
    if Hb > 0 and taper_mag > 0:
        bore_top_radius = (Db / 2.0) - Hb * math.tan(math.radians(taper_mag))
        if bore_top_radius < MIN_BORE_TOP_RADIUS:
            # solve for the max taper magnitude that keeps the hole open
            max_taper_rad = math.atan(
                max((Db / 2.0) - MIN_BORE_TOP_RADIUS, 0.0) / Hb
            )
            max_taper_deg = math.degrees(max_taper_rad)
            taper = -max_taper_deg if taper < 0 else max_taper_deg
            clamped_flags.append("taper_bore_closing")

    # --- bolt circle diameter: midpoint between boss OD and flange OD ---
    BCD = (boss_od + D) / 2.0
    R = BCD / 2.0

    # --- bolt hole diameter clamp (unchanged from v2) ---
    max_dh_by_boss = 2.0 * (R - boss_od / 2.0 - MIN_HOLE_CLEARANCE)
    max_dh_by_rim = 2.0 * (D / 2.0 - R - MIN_HOLE_CLEARANCE)
    max_dh = max(4.0, min(max_dh_by_boss, max_dh_by_rim))
    if dh > max_dh:
        dh = max_dh
        clamped_flags.append("dh")

    fillet_large = clip(0.10 * T, 2.0, 8.0)
    fillet_small = clip(0.05 * T, 1.0, 4.0)

    bore_depth = T
    boss_hole_depth = Hb

    return {
        "Db": round(Db, 3),
        "boss_od": round(boss_od, 3),
        "BCD": round(BCD, 3),
        "dh": round(dh, 3),
        "taper": round(taper, 3),
        "fillet_large": round(fillet_large, 3),
        "fillet_small": round(fillet_small, 3),
        "bore_depth": round(bore_depth, 3),
        "boss_hole_depth": round(boss_hole_depth, 3),
        "clamped": ";".join(clamped_flags) if clamped_flags else "none",
    }


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main():
    unit_samples = lhs(N_SAMPLES, len(VAR_ORDER), seed=SEED)
    scaled = scale_to_bounds(unit_samples, VAR_ORDER, BOUNDS)

    rows = []
    for i in range(N_SAMPLES):
        indep = {name: float(scaled[i, j]) for j, name in enumerate(VAR_ORDER)}
        dep = compute_dependent(scaled[i, :])

        row = {"sample_id": i + 1}
        row["D"] = round(indep["D"], 4)
        row["T"] = round(indep["T"], 4)
        row["bore_ratio"] = round(indep["bore_ratio"], 4)
        row["wall_t"] = round(indep["wall_t"], 4)
        row["Hb"] = round(indep["Hb"], 4)
        row["taper"] = dep["taper"]            # possibly clamped
        row["Db"] = dep["Db"]
        row["boss_od"] = dep["boss_od"]
        row["BCD"] = dep["BCD"]
        row["dh"] = dep["dh"]
        row["fillet_large"] = dep["fillet_large"]
        row["fillet_small"] = dep["fillet_small"]
        row["bore_depth"] = dep["bore_depth"]
        row["boss_hole_depth"] = dep["boss_hole_depth"]
        row["clamped"] = dep["clamped"]
        rows.append(row)

    fieldnames = ["sample_id", "D", "T", "bore_ratio", "wall_t", "Hb", "taper",
                  "Db", "boss_od", "BCD", "dh",
                  "fillet_large", "fillet_small",
                  "bore_depth", "boss_hole_depth", "clamped"]

    out_path = os.path.abspath(OUT_CSV)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    n_clamped = sum(1 for r in rows if r["clamped"] != "none")
    n_taper_fix = sum(1 for r in rows if "taper_bore_closing" in r["clamped"])
    print("DOE generated: {} samples written to {}".format(N_SAMPLES, out_path))
    if n_taper_fix:
        print("Note: {} sample(s) had taper reduced to keep the hub bore open "
              "(see 'clamped' column: taper_bore_closing).".format(n_taper_fix))
    if n_clamped:
        print("Total samples with any clamping: {}".format(n_clamped))
    else:
        print("No clamping needed - all samples respected safety margins natively.")


if __name__ == "__main__":
    main()