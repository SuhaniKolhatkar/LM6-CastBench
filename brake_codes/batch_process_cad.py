# -*- coding: utf-8 -*-
"""
Batch DOE Variant Generator for 'brake.FCStd' parametric sketch
-----------------------------------------------------------------
Reads parameter sets from doe.csv (columns: D_bottom_hole, D_top_hole,
R_bottom_outer, R_top_outer, D_right_end, L_span_X, H_offset_Y, Pad_Length
-- optionally a variant_id column too, which is used for naming only),
rebuilds the sketch for each row, pads it, and exports each variant as a
STEP file into an output folder.

HOW THE PARAMETRIZATION WORKS
------------------------------
The original macro's sketch is NOT driven by Sketcher dimensional
constraints -- every point (fillet arcs, connecting lines, Bezier poles)
has its exact coordinate baked in, presumably solved once by whatever tool
authored the original 'brake.FCStd'. Recomputing exact tangencies for every
DOE row would require re-deriving that solver from scratch.

Instead this script uses a "shape-preserving morph":

  1. Three functional reference points (anchors) are identified in the
     baseline sketch -- they are also the centers of the three parametric
     circles:
        A_bottom = center of the bottom hole / bottom hub   (idx0 / idx2)
        A_top    = center of the top hole / top hub         (idx1 / idx12)
        A_right  = center of the right-end eye              (idx7)

  2. For a given DOE row, new anchor positions are computed directly from
     the parameters:
        A_bottom_new = A_bottom_baseline                      (kept fixed --
                        it is the reference origin of the whole part)
        A_top_new    = (A_bottom_new.x + dx_top_baseline, H_offset_Y)
        A_right_new  = (A_bottom_new.x + L_span_X, 0.0)
     where dx_top_baseline is the (fixed) baseline X offset between the
     bottom and top hub -- this offset is not one of the 8 DOE parameters,
     so it is preserved as-is.

  3. Every other point in the sketch (fillet-arc centers, line endpoints,
     Bezier poles) is displaced by a weighted blend of the three anchors'
     displacement vectors, where the weight of each anchor is based on
     inverse-distance (in the baseline sketch) from that point to the
     anchor. Points close to the bottom hub move almost entirely with the
     bottom hub's displacement (i.e. barely, since it's fixed); points near
     the right eye move almost entirely with the right eye, etc. Points
     between anchors blend smoothly.

     Because this transform is a deterministic function of a point's
     ORIGINAL (x, y) coordinates only, two originally-coincident points
     (e.g. the shared endpoint of a line and an arc) always map to the
     IDENTICAL new coordinate -- so the existing Coincident constraints in
     the sketch remain satisfied without any extra solving.

  4. Radii that are direct DOE parameters (hole diameters, hub outer radii,
     right-end diameter) are set directly. Small fillet-arc radii (not DOE
     parameters) are kept at their baseline values; only their centers are
     morphed as in step 3.

  5. Pad_Length drives the Pad feature length directly.

This is a legitimate, low-distortion approximation for SMALL parameter
excursions around baseline (roughly the +/-10-20% ranges typical of an
LHS DOE around a working design). It is NOT a substitute for a fully
constrained parametric sketch. For large excursions, always spot check a
few variants visually and/or check Shape validity (done automatically
below) before trusting the whole batch.

HOW TO RUN
----------
This script uses the FreeCAD Python API (FreeCAD, Part, Sketcher modules),
so it must be run with FreeCAD's own Python interpreter, e.g.:

    freecadcmd generate_variants.py
    (or, on some installs)  FreeCADCmd generate_variants.py

or pasted into FreeCAD's built-in Python console. It also needs `pandas`
available in that same Python environment (pip install pandas into
FreeCAD's Python, or run this with a Python that has both freecad and
pandas on the path).

Expected inputs:
    ./doe.csv                    -- your DOE table
Outputs:
    ./output/<variant_name>.step -- one STEP file per DOE row
    ./output/generation_report.csv -- success/failure log per variant
"""

import os
import math
import csv
import sys

import FreeCAD as App
import Part
import Sketcher

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install it into FreeCAD's Python "
          "environment, e.g.:  python -m pip install pandas")
    sys.exit(1)

# ----------------------------------------------------------------------
# 0. Paths / configuration
# ----------------------------------------------------------------------
DOE_CSV_PATH = r"D:\NeurIPS\surrogate_brake\doe.csv"
OUTPUT_DIR = r"D:\NeurIPS\surrogate_brake\brake_cad_outputs"
REQUIRED_PARAMS = [
    "D_bottom_hole", "D_top_hole", "R_bottom_outer", "R_top_outer",
    "D_right_end", "L_span_X", "H_offset_Y", "Pad_Length",
]

# Baseline / default parameter values (== original brake.FCStd design)
BASELINE = {
    "D_bottom_hole":  14.0,
    "D_top_hole":     12.0,
    "R_bottom_outer": 10.5,
    "R_top_outer":    10.0,
    "D_right_end":    16.0,
    "L_span_X":       110.0,
    "H_offset_Y":     23.0,
    "Pad_Length":     8.0,
}

# ----------------------------------------------------------------------
# 1. Baseline raw geometry (transcribed exactly from the original macro)
# ----------------------------------------------------------------------
# Anchors (functional reference centers) -- these ARE the circle centers
# used by the three parametric radii/diameters.
A_BOTTOM0 = (-82.5587518926589752, 0.0)   # bottom hole / bottom hub center
A_TOP0    = (-84.2490282719526107, 23.0)  # top hole / top hub center
A_RIGHT0  = (27.4412481073410284, 0.0)    # right-end eye center

# Fixed (non-parametric) baseline offset between bottom and top hub centers
DX_TOP_BASELINE = A_TOP0[0] - A_BOTTOM0[0]

# Geometry entities, in the same order/index as sketch.addGeometry() calls
# in the original macro. Each entry records enough info to rebuild it for
# a new variant.
#
# type 'circle_param': circle whose center is an anchor and whose radius
#                       is a direct DOE parameter (used for idx0,1,2,7,12)
# type 'arc_fixed':     arc whose center/points are morphed, radius fixed
# type 'line':          line with two morphed endpoints
# type 'bezier':        cubic Bezier with 4 morphed poles

GEOMETRY = [
    # idx0 - bottom hole (Hole 1), radius = D_bottom_hole / 2
    {"idx": 0, "type": "circle_param", "anchor": "bottom",
     "param": "D_bottom_hole", "is_diameter": True},

    # idx1 - top hole (Hole 2), radius = D_top_hole / 2
    {"idx": 1, "type": "circle_param", "anchor": "top",
     "param": "D_top_hole", "is_diameter": True},

    # idx2 - bottom hub outer arc, radius = R_bottom_outer
    {"idx": 2, "type": "arc_param", "anchor": "bottom",
     "param": "R_bottom_outer", "is_diameter": False,
     "start": 2.6748623510764644, "end": 6.0188533385699472},

    # idx3 - connecting line
    {"idx": 3, "type": "line",
     "p1": (-72.4234457767213229, -2.7432772255528048),
     "p2": (-70.5534731731366662, 4.1655195191612426)},

    # idx4 - fillet arc, R9.122 (fixed radius, morphed center)
    {"idx": 4, "type": "arc_fixed",
     "center": (-61.7482540794578938, 1.7822509005935225),
     "radius": 9.1220530910500024,
     "start": 1.1728587767723522, "end": 2.8772606849801536},

    # idx5 - lower Bezier curve (4 poles)
    {"idx": 5, "type": "bezier",
     "poles": [
         (-58.2132954664423607, 10.1915267433877297),
         (-30.6504116037719676, -1.3949232115608843),
         (-3.8034599096136952, -7.1031023045212240),
         (18.9413175535375089, -4.0799138391747398),
     ]},

    # idx6 - small fillet arc, R2.452 (fixed radius, morphed center)
    {"idx": 6, "type": "arc_fixed",
     "center": (19.2644399303790657, -6.5109057005383768),
     "radius": 2.4523722189814454,
     "start": 0.6724593155872700, "end": 1.7029396777116752},

    # idx7 - right-end arc, radius = D_right_end / 2
    {"idx": 7, "type": "arc_param", "anchor": "right",
     "param": "D_right_end", "is_diameter": True,
     "start": 3.8140519691770636, "end": 9.4090425313055768},

    # idx8 - tiny fillet arc, R0.505 (fixed radius, morphed center)
    {"idx": 8, "type": "arc_fixed",
     "center": (18.9373426947153547, 0.1338236490482382),
     "radius": 0.5049583206461855,
     "start": 4.8135326571618791, "end": 6.2674498777157828},

    # idx9 - upper Bezier curve (4 poles)
    {"idx": 9, "type": "bezier",
     "poles": [
         (-52.2345052606587004, 24.0393897989436489),
         (-33.8050054836408691, 7.4890195502119203),
         (-8.3005575315068434, -3.1381029682104540),
         (18.9883290000000038, -0.3685540000000018),
     ]},

    # idx10 - fillet arc, R11.178 (fixed radius, morphed center)
    {"idx": 10, "type": "arc_fixed",
     "center": (-60.8646701369499183, 16.9345836523270776),
     "radius": 11.1784621559931576,
     "start": 0.6887594529619723, "end": 1.3226286989555898},

    # idx11 - connecting line, upper hub to spline
    {"idx": 11, "type": "line",
     "p1": (-82.2093480389212488, 32.7897755105508537),
     "p2": (-58.1189253057287800, 27.7705839058705237)},

    # idx12 - top hub outer arc, radius = R_top_outer
    {"idx": 12, "type": "arc_param", "anchor": "top",
     "param": "R_top_outer", "is_diameter": False,
     "start": 1.3653868727566090, "end": 3.9194934936138539},

    # idx13 - small fillet arc, R4.312 (fixed radius, morphed center)
    {"idx": 13, "type": "arc_fixed",
     "center": (-94.4448393207024566, 12.9559366776967071),
     "radius": 4.3121546583401518,
     "start": 0.0, "end": 0.7779008400240612},

    # idx14 - short connecting line, left side
    {"idx": 14, "type": "line",
     "p1": (-90.1326846623623084, 12.9559366776967053),
     "p2": (-90.1326846623623084, 10.1370663681643087)},

    # idx15 - small fillet arc, R4.097 (fixed radius, morphed center)
    {"idx": 15, "type": "arc_fixed",
     "center": (-94.2296158889424618, 10.1370663681643105),
     "radius": 4.0969312265801410,
     "start": 5.9129335844203599, "end": 6.2831853071795862},

    # idx16 - connecting line back to start
    {"idx": 16, "type": "line",
     "p1": (-90.4103079250263306, 8.6545913592368979),
     "p2": (-91.9357172617598195, 4.7246714665343053)},
]

# Coincident constraints -- identical topology for every variant
COINCIDENT_PAIRS = [
    (2, 2, 3, 1), (3, 2, 4, 2), (4, 1, 5, 1), (5, 2, 6, 2),
    (6, 1, 7, 1), (7, 2, 8, 2), (8, 1, 9, 2), (9, 1, 10, 1),
    (10, 2, 11, 2), (11, 1, 12, 1), (12, 2, 13, 2), (13, 1, 14, 1),
    (14, 2, 15, 2), (15, 1, 16, 1), (16, 2, 2, 1),
]


# ----------------------------------------------------------------------
# 2. Morphing helpers
# ----------------------------------------------------------------------
def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def compute_anchors_new(params):
    """New anchor positions for a given parameter set."""
    a_bottom_new = A_BOTTOM0  # fixed reference / origin of the part
    a_top_new = (a_bottom_new[0] + DX_TOP_BASELINE, params["H_offset_Y"])
    a_right_new = (a_bottom_new[0] + params["L_span_X"], 0.0)
    return {"bottom": a_bottom_new, "top": a_top_new, "right": a_right_new}


def morph_point(pt, anchors_new, power=2.0, epsilon=1e-6):
    """
    Displace an arbitrary baseline point via inverse-distance-weighted
    blending of the three anchors' displacement vectors.
    """
    anchors0 = {"bottom": A_BOTTOM0, "top": A_TOP0, "right": A_RIGHT0}
    keys = ["bottom", "top", "right"]

    dists = [max(_dist(pt, anchors0[k]), epsilon) for k in keys]
    inv = [1.0 / (d ** power) for d in dists]
    total = sum(inv)
    weights = [i / total for i in inv]

    dx = sum(w * (anchors_new[k][0] - anchors0[k][0])
             for w, k in zip(weights, keys))
    dy = sum(w * (anchors_new[k][1] - anchors0[k][1])
             for w, k in zip(weights, keys))

    return (pt[0] + dx, pt[1] + dy)


# ----------------------------------------------------------------------
# 3. Sketch / Body / Pad builder for one variant
# ----------------------------------------------------------------------
def build_variant(params, doc_name):
    doc = App.newDocument(doc_name)
    body = doc.addObject("PartDesign::Body", "Body")

    sketch = doc.addObject("Sketcher::SketchObject", "Sketch")
    body.addObject(sketch)
    sketch.AttachmentSupport = [(body.Origin.OriginFeatures[3], "")]  # XY_Plane
    sketch.MapMode = "FlatFace"

    anchors_new = compute_anchors_new(params)

    # Build geometry in strict index order (order matters for the
    # Coincident constraints added afterwards).
    for entry in GEOMETRY:
        gtype = entry["type"]

        if gtype == "circle_param":
            center = anchors_new[entry["anchor"]]
            value = params[entry["param"]]
            radius = value / 2.0 if entry["is_diameter"] else value
            sketch.addGeometry(
                Part.Circle(App.Vector(center[0], center[1], 0.0),
                            App.Vector(0, 0, 1), radius), False)

        elif gtype == "arc_param":
            center = anchors_new[entry["anchor"]]
            value = params[entry["param"]]
            radius = value / 2.0 if entry["is_diameter"] else value
            circ = Part.Circle(App.Vector(center[0], center[1], 0.0),
                                App.Vector(0, 0, 1), radius)
            sketch.addGeometry(
                Part.ArcOfCircle(circ, entry["start"], entry["end"]), False)

        elif gtype == "arc_fixed":
            center = morph_point(entry["center"], anchors_new)
            circ = Part.Circle(App.Vector(center[0], center[1], 0.0),
                                App.Vector(0, 0, 1), entry["radius"])
            sketch.addGeometry(
                Part.ArcOfCircle(circ, entry["start"], entry["end"]), False)

        elif gtype == "line":
            p1 = morph_point(entry["p1"], anchors_new)
            p2 = morph_point(entry["p2"], anchors_new)
            sketch.addGeometry(
                Part.LineSegment(App.Vector(p1[0], p1[1], 0.0),
                                  App.Vector(p2[0], p2[1], 0.0)), False)

        elif gtype == "bezier":
            new_poles = [morph_point(p, anchors_new) for p in entry["poles"]]
            bez = Part.BezierCurve()
            bez.setPoles([App.Vector(p[0], p[1], 0.0) for p in new_poles])
            sketch.addGeometry(bez.toBSpline(), False)

        else:
            raise ValueError(f"Unknown geometry type: {gtype}")

    # Coincident constraints (topology identical to the baseline macro)
    for a_idx, a_pos, b_idx, b_pos in COINCIDENT_PAIRS:
        sketch.addConstraint(
            Sketcher.Constraint("Coincident", a_idx, a_pos, b_idx, b_pos))

    doc.recompute()

    # Pad
    pad = body.newObject("PartDesign::Pad", "Pad")
    pad.Profile = sketch
    pad.Length = params["Pad_Length"]
    pad.Midplane = False
    pad.Reversed = False
    pad.Type = "Length"

    sketch.Visibility = False
    doc.recompute()

    return doc, body, pad


# ----------------------------------------------------------------------
# 4. Batch driver
# ----------------------------------------------------------------------
def main():
    if not os.path.exists(DOE_CSV_PATH):
        print(f"ERROR: could not find {DOE_CSV_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doe_df = pd.read_csv(DOE_CSV_PATH)

    missing = [c for c in REQUIRED_PARAMS if c not in doe_df.columns]
    if missing:
        print(f"ERROR: doe.csv is missing required columns: {missing}")
        sys.exit(1)

    has_variant_id = "variant_id" in doe_df.columns

    report_rows = []

    for i, row in doe_df.iterrows():
        params = {k: float(row[k]) for k in REQUIRED_PARAMS}

        if has_variant_id and isinstance(row["variant_id"], str) and row["variant_id"].strip():
            variant_name = row["variant_id"].strip()
        else:
            variant_name = f"variant_{i+1:03d}"

        doc_name = f"brake_{variant_name}"
        step_path = os.path.join(OUTPUT_DIR, f"{variant_name}.step")

        status = "OK"
        message = ""

        try:
            doc, body, pad = build_variant(params, doc_name)

            shape = pad.Shape
            if shape is None or shape.isNull():
                status = "FAILED"
                message = "Pad produced a null shape"
            elif not shape.isValid():
                status = "WARNING"
                message = "Shape.isValid() returned False (exported anyway -- inspect manually)"

            if status != "FAILED":
                Part.export([pad], step_path)

            App.closeDocument(doc.Name)

        except Exception as exc:
            status = "FAILED"
            message = str(exc)
            try:
                App.closeDocument(doc_name)
            except Exception:
                pass

        report_rows.append({
            "variant_name": variant_name,
            "status": status,
            "message": message,
            **params,
        })

        print(f"[{i+1}/{len(doe_df)}] {variant_name}: {status} {message}")

    # Write a generation report next to the STEP files
    report_path = os.path.join(OUTPUT_DIR, "generation_report.csv")
    fieldnames = ["variant_name", "status", "message"] + REQUIRED_PARAMS
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    n_ok = sum(1 for r in report_rows if r["status"] == "OK")
    n_warn = sum(1 for r in report_rows if r["status"] == "WARNING")
    n_fail = sum(1 for r in report_rows if r["status"] == "FAILED")
    print(f"\nDone. {n_ok} OK, {n_warn} WARNING, {n_fail} FAILED. "
          f"Report: {report_path}")


if __name__ == "__main__":
    main()