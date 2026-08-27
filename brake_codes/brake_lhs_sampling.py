"""
LHS DOE Generator for Impeller CAD Parametric Model
-----------------------------------------------------
Generates a Latin Hypercube Sampling (LHS) based Design of Experiments (DOE)
table for the parametric CAD model, centered around the baseline geometry:

    D_bottom_hole   = 14
    D_top_hole      = 12
    R_bottom_outer  = 10.5
    R_top_outer     = 10.0
    D_right_end     = 16
    L_span_X        = 110
    H_offset_Y      = 23
    Pad_Length      = 8

The variation ranges below are deliberately kept tight (roughly +/-10-20% of
baseline) and geometric feasibility constraints are enforced so that the
sampled parameter combinations do NOT break the CAD model (e.g. a hole
diameter larger than the outer diameter it sits in, negative wall thickness,
top hole bigger than bottom hole for a tapered/stepped bore, etc.).

Requires: numpy, scipy (>=1.7 for scipy.stats.qmc), pandas
Install if needed:
    pip install numpy scipy pandas
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------

N_SAMPLES = 10          # number of DOE points to generate
RANDOM_SEED = 42         # set to None for a non-reproducible sample
MIN_WALL_THICKNESS = 1.5  # minimum allowable wall thickness (mm) between a
                          # hole diameter and its surrounding outer radius,
                          # used as a feasibility constraint below

# Parameter bounds: (lower_bound, upper_bound)
# Centered on baseline values, with a sensible spread that keeps the
# geometry valid for the CAD generator.
PARAM_BOUNDS = {
    "D_bottom_hole":  (12.0, 16.0),   # baseline 14
    "D_top_hole":     (10.0, 14.0),   # baseline 12
    "R_bottom_outer": (9.5, 11.5),    # baseline 10.5
    "R_top_outer":    (9.0, 11.0),    # baseline 10.0
    "D_right_end":    (14.0, 18.0),   # baseline 16
    "L_span_X":       (95.0, 125.0),  # baseline 110
    "H_offset_Y":     (18.0, 28.0),   # baseline 23
    "Pad_Length":     (6.0, 12.0),    # baseline 8
}

# Decimal precision to round each parameter to (keeps CAD inputs clean)
ROUND_DECIMALS = {
    "D_bottom_hole":  1,
    "D_top_hole":     1,
    "R_bottom_outer": 2,
    "R_top_outer":    2,
    "D_right_end":    1,
    "L_span_X":       1,
    "H_offset_Y":     1,
    "Pad_Length":     1,
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
LOWER = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
UPPER = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])


# ----------------------------------------------------------------------
# 2. Feasibility constraints
# ----------------------------------------------------------------------
def is_feasible(row: dict) -> bool:
    """
    Returns True if a sampled parameter combination is geometrically valid
    for the CAD model. Add/adjust rules here as the CAD model's actual
    constraints become clearer.
    """
    D_bottom_hole = row["D_bottom_hole"]
    D_top_hole = row["D_top_hole"]
    R_bottom_outer = row["R_bottom_outer"]
    R_top_outer = row["R_top_outer"]
    D_right_end = row["D_right_end"]
    L_span_X = row["L_span_X"]
    H_offset_Y = row["H_offset_Y"]
    Pad_Length = row["Pad_Length"]

    # (a) Bottom hole must be smaller than its outer diameter,
    #     leaving at least MIN_WALL_THICKNESS of solid wall.
    if D_bottom_hole > 2 * R_bottom_outer - 2 * MIN_WALL_THICKNESS:
        return False

    # (b) Top hole must be smaller than its outer diameter,
    #     leaving at least MIN_WALL_THICKNESS of solid wall.
    if D_top_hole > 2 * R_top_outer - 2 * MIN_WALL_THICKNESS:
        return False

    # (c) Maintain a tapered/stepped bore: bottom hole >= top hole
    #     (keeps the hole profile monotonic, avoids an undercut the
    #     CAD kernel can fail to resolve).
    if D_top_hole > D_bottom_hole:
        return False

    # (d) Maintain a tapered/stepped outer profile: bottom outer radius
    #     should not be smaller than the top outer radius.
    if R_top_outer > R_bottom_outer:
        return False

    # (e) D_right_end must clear the largest outer diameter, otherwise
    #     the end feature collides with the body.
    if D_right_end < 2 * R_bottom_outer * 0.9:
        return False

    # (f) Pad length should stay well below the overall span so the pad
    #     feature does not consume the whole body.
    if Pad_Length > 0.15 * L_span_X:
        return False

    # (g) H_offset_Y sanity: keep offset comfortably below half the span
    #     to avoid the feature moving outside the body envelope.
    if H_offset_Y > 0.3 * L_span_X:
        return False

    return True


# ----------------------------------------------------------------------
# 3. LHS sampling with rejection of infeasible points
# ----------------------------------------------------------------------
def generate_doe(n_samples: int, seed: int = None) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)

    accepted_rows = []
    batch_size = max(n_samples * 4, 50)  # over-sample to account for rejections
    max_attempts = 50
    attempt = 0

    while len(accepted_rows) < n_samples and attempt < max_attempts:
        attempt += 1
        unit_samples = sampler.random(n=batch_size)
        scaled = qmc.scale(unit_samples, LOWER, UPPER)

        for sample in scaled:
            row = {name: val for name, val in zip(PARAM_NAMES, sample)}
            # round to the configured precision
            row = {name: round(val, ROUND_DECIMALS[name]) for name, val in row.items()}

            if is_feasible(row):
                accepted_rows.append(row)
            if len(accepted_rows) >= n_samples:
                break

    if len(accepted_rows) < n_samples:
        raise RuntimeError(
            f"Could only generate {len(accepted_rows)} feasible samples out of "
            f"{n_samples} requested after {max_attempts} attempts. "
            f"Consider loosening PARAM_BOUNDS or the feasibility constraints."
        )

    df = pd.DataFrame(accepted_rows, columns=PARAM_NAMES)
    return df


# ----------------------------------------------------------------------
# 4. Run and save
# ----------------------------------------------------------------------
if __name__ == "__main__":
    doe_df = generate_doe(N_SAMPLES, seed=RANDOM_SEED)

    output_path = "doe_lhs_samples.csv"
    doe_df.to_csv(output_path, index=False)

    print(f"Generated {len(doe_df)} feasible LHS DOE samples.")
    print(f"Saved to: {output_path}\n")
    print(doe_df.to_string(index=False))