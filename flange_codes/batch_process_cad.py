# -*- coding: utf-8 -*-
"""
batch_process_cad.py  (v3 - geometry-based fillets, robust)

Reads doe.csv and generates flange variants.

KEY CHANGES vs v2:
1. Bolt holes (Pocket001) now use Type=1 (ThroughAll) so they always
   cut through the ENTIRE part (boss + base plate), not just the boss.
2. Fillets are no longer PartDesign::Fillet features referencing
   hardcoded edge names (Edge11, Edge2, ...). Those break whenever
   topology shifts. Instead:
     - The body is built through Pocket001 with NO fillets.
     - The final raw shape is post-processed with Part.Shape.makeFillet(),
       selecting edges purely by GEOMETRY (radius + Z-height), which is
       stable regardless of feature/edge renumbering.
     - Filleting is attempted in two radius groups (large/small). If a
       batch fillet fails (OCC rejects it), it falls back to filleting
       edges one at a time and keeps whichever succeed - so the part
       always comes out filleted to the maximum extent possible,
       instead of the whole variant failing.

Run with:
    freecadcmd batch_process_cad.py
"""

import FreeCAD as App
import Part
import Sketcher
import csv
import os
import math

DOE_CSV = r"D:\NeurIPS\surrogate_flange\doe.csv"
OUT_DIR = r"D:\NeurIPS\surrogate_flange\flange_cad_outputs"


# ===================================================================
# Geometry-based fillet helpers
# ===================================================================
def circle_edges(shape, r_target, z_target, r_tol=1.0, z_tol=1.0):
    """Find edges on `shape` that are circular arcs with the given
    radius and Z-height (within tolerance)."""
    found = []
    for e in shape.Edges:
        try:
            c = e.Curve
        except Exception:
            continue
        if not hasattr(c, "Radius"):
            continue
        try:
            r = c.Radius
            z = e.CenterOfMass.z
        except Exception:
            continue
        if abs(r - r_target) <= r_tol and abs(z - z_target) <= z_tol:
            found.append(e)
    return found


def dedupe_edges(edges):
    seen = set()
    out = []
    for e in edges:
        try:
            com = e.CenterOfMass
            key = (round(com.x, 2), round(com.y, 2), round(com.z, 2), round(e.Length, 2))
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def locate_matching_edge(shape, ref_edge, tol=0.75):
    """Find the edge in `shape` geometrically matching `ref_edge`
    (needed because edges from an old shape can't be passed directly
    into makeFillet on a newer shape)."""
    try:
        ref_com = ref_edge.CenterOfMass
        ref_len = ref_edge.Length
    except Exception:
        return None
    for e in shape.Edges:
        try:
            com = e.CenterOfMass
            if com.distanceToPoint(ref_com) <= tol and abs(e.Length - ref_len) <= tol:
                return e
        except Exception:
            continue
    return None


def safe_fillet(shape, edges, radius, label=""):
    """Try filleting all edges at once. If that fails, fall back to
    filleting one edge at a time, keeping whichever succeed."""
    if not edges:
        return shape, 0, 0

    try:
        new_shape = shape.makeFillet(radius, edges)
        return new_shape, len(edges), 0
    except Exception:
        pass

    current = shape
    applied = 0
    failed = 0
    for ref_edge in edges:
        match = locate_matching_edge(current, ref_edge)
        if match is None:
            failed += 1
            continue
        try:
            current = current.makeFillet(radius, [match])
            applied += 1
        except Exception:
            failed += 1
            continue

    if failed:
        print("    [{}] {} edge(s) could not be filleted and were skipped".format(label, failed))

    return current, applied, failed


def fillet_flange_shape(shape, D, T, Db, boss_od, Hb, taper, dh, R,
                          fillet_large, fillet_small):
    """
    Applies fillets to the raw flange shape by locating edges geometrically:
      LARGE radius group:
        - outer top rim of base plate      (r = D/2,        z = T)
        - step where boss meets base top   (r = boss_od/2,  z = T)
        - top rim of boss                  (r = boss_top_r, z = T+Hb)
      SMALL radius group:
        - bolt hole top edges (x4)         (r = dh/2,       z = T+Hb)
    """
    taper_rad = math.radians(taper)
    boss_top_r = (boss_od / 2.0) + Hb * math.tan(taper_rad)
    boss_top_r = max(boss_top_r, 1.0)

    large_candidates = []
    large_candidates += circle_edges(shape, D / 2.0, T, r_tol=1.0, z_tol=1.0)
    large_candidates += circle_edges(shape, boss_od / 2.0, T, r_tol=1.0, z_tol=1.0)
    large_candidates += circle_edges(shape, boss_top_r, T + Hb, r_tol=1.5, z_tol=1.0)
    large_candidates = dedupe_edges(large_candidates)

    result_shape, n_ok, n_fail = safe_fillet(shape, large_candidates, fillet_large, label="large")
    total_applied = n_ok
    total_failed = n_fail

    # re-identify small edges on the UPDATED shape (never trust old indices/refs)
    small_candidates = circle_edges(result_shape, dh / 2.0, T + Hb, r_tol=0.75, z_tol=1.0)
    small_candidates = dedupe_edges(small_candidates)

    result_shape, n_ok2, n_fail2 = safe_fillet(result_shape, small_candidates, fillet_small, label="small")
    total_applied += n_ok2
    total_failed += n_fail2

    return result_shape, total_applied, total_failed


# ===================================================================
# Body construction (PartDesign, NO fillets here)
# ===================================================================
def build_variant(row, variant_id):
    D   = float(row["D"])
    T   = float(row["T"])
    Db  = float(row["Db"])
    Hb  = float(row["Hb"])
    taper = float(row["taper"])
    dh  = float(row["dh"])
    boss_od = float(row["boss_od"])
    BCD = float(row["BCD"])
    fillet_large = float(row["fillet_large"])
    fillet_small = float(row["fillet_small"])
    bore_depth = float(row["bore_depth"])

    R = BCD / 2.0

    doc_name = "Flange_{}".format(variant_id)
    doc = App.newDocument(doc_name)
    body = doc.addObject('PartDesign::Body', 'Body')
    doc.recompute()

    # ---- Sketch (base disc) -> Pad ----
    sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
    sketch.AttachmentSupport = (doc.getObject('XY_Plane'), [''])
    sketch.MapMode = 'FlatFace'
    doc.recompute()
    sketch.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), D / 2.0), False)
    sketch.addConstraint(Sketcher.Constraint('Diameter', 0, D))
    sketch.addConstraint(Sketcher.Constraint('Coincident', 0, 3, -1, 1))
    doc.recompute()

    pad = body.newObject('PartDesign::Pad', 'Pad')
    pad.Profile = (sketch, [''])
    pad.Length = T
    pad.TaperAngle = 0.0
    pad.Direction = (0, 0, 1)
    pad.ReferenceAxis = (sketch, ['N_Axis'])
    pad.Type = 0
    sketch.Visibility = False
    doc.recompute()

    # ---- Sketch001 (bore) -> Pocket (through base) ----
    sketch001 = body.newObject('Sketcher::SketchObject', 'Sketch001')
    sketch001.AttachmentSupport = (pad, ['Face3'])
    sketch001.MapMode = 'FlatFace'
    doc.recompute()
    sketch001.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), Db / 2.0), False)
    sketch001.addConstraint(Sketcher.Constraint('Diameter', 0, Db))
    sketch001.addConstraint(Sketcher.Constraint('Coincident', 0, 3, -1, 1))
    doc.recompute()

    pocket = body.newObject('PartDesign::Pocket', 'Pocket')
    pocket.Profile = (sketch001, [''])
    pocket.Type = 1
    pocket.Direction = (0, 0, -1)
    pocket.ReferenceAxis = (sketch001, ['N_Axis'])
    sketch001.Visibility = False
    pad.Visibility = False
    doc.recompute()

    # ---- Sketch002 (annulus) -> Pad001 (tapered boss) ----
    sketch002 = body.newObject('Sketcher::SketchObject', 'Sketch002')
    sketch002.AttachmentSupport = (pocket, ['Face3'])
    sketch002.MapMode = 'FlatFace'
    doc.recompute()
    sketch002.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), Db / 2.0), False)
    sketch002.addConstraint(Sketcher.Constraint('Diameter', 0, Db))
    sketch002.addConstraint(Sketcher.Constraint('Coincident', 0, 3, -1, 1))
    sketch002.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), boss_od / 2.0), False)
    sketch002.addConstraint(Sketcher.Constraint('Diameter', 1, boss_od))
    sketch002.addConstraint(Sketcher.Constraint('Coincident', 1, 3, 0, 3))
    doc.recompute()

    pad001 = body.newObject('PartDesign::Pad', 'Pad001')
    pad001.Profile = (sketch002, [''])
    pad001.Length = Hb
    pad001.TaperAngle = taper
    pad001.Direction = (0, 0, 1)
    pad001.ReferenceAxis = (sketch002, ['N_Axis'])
    pad001.Type = 0
    sketch002.Visibility = False
    pocket.Visibility = False
    doc.recompute()

    # ---- Sketch003 (4x bolt holes) -> Pocket001 (THROUGH ALL) ----
    sketch003 = body.newObject('Sketcher::SketchObject', 'Sketch003')
    sketch003.AttachmentSupport = (pad001, ['Face3'])
    sketch003.MapMode = 'FlatFace'
    doc.recompute()

    dh_half = dh / 2.0
    sketch003.addGeometry(Part.Circle(App.Vector(0.0, R, 0.0), App.Vector(0, 0, 1), dh_half), False)
    sketch003.addConstraint(Sketcher.Constraint('Diameter', 0, dh))
    sketch003.addConstraint(Sketcher.Constraint('PointOnObject', 0, 3, -2))
    sketch003.addConstraint(Sketcher.Constraint('DistanceY', -1, 1, 0, 3, R))

    sketch003.addGeometry(Part.Circle(App.Vector(R, 0.0, 0.0), App.Vector(0, 0, 1), dh_half), False)
    sketch003.addConstraint(Sketcher.Constraint('Diameter', 1, dh))
    sketch003.addConstraint(Sketcher.Constraint('PointOnObject', 1, 3, -1))
    sketch003.addConstraint(Sketcher.Constraint('DistanceX', -1, 1, 1, 3, R))

    sketch003.addGeometry(Part.Circle(App.Vector(0.0, -R, 0.0), App.Vector(0, 0, 1), dh_half), False)
    sketch003.addConstraint(Sketcher.Constraint('Diameter', 2, dh))
    sketch003.addConstraint(Sketcher.Constraint('PointOnObject', 2, 3, -2))
    sketch003.addConstraint(Sketcher.Constraint('DistanceY', 2, 3, -1, 1, R))

    sketch003.addGeometry(Part.Circle(App.Vector(-R, 0.0, 0.0), App.Vector(0, 0, 1), dh_half), False)
    sketch003.addConstraint(Sketcher.Constraint('Diameter', 3, dh))
    sketch003.addConstraint(Sketcher.Constraint('PointOnObject', 3, 3, -1))
    sketch003.addConstraint(Sketcher.Constraint('DistanceX', 3, 3, -1, 1, R))
    doc.recompute()

    pocket001 = body.newObject('PartDesign::Pocket', 'Pocket001')
    pocket001.Profile = (sketch003, [''])
    pocket001.Type = 1          # <-- ThroughAll: goes through boss AND base plate
    pocket001.Direction = (0, 0, -1)
    pocket001.ReferenceAxis = (sketch003, ['N_Axis'])
    sketch003.Visibility = False
    pad001.Visibility = False
    doc.recompute()

    body.Tip = pocket001
    doc.recompute()

    # ---- NO PartDesign fillets here. Return raw shape for post-processing. ----
    raw_shape = pocket001.Shape.copy()
    return doc, body, raw_shape


def export_step(shape, variant_id, out_dir):
    filename = "flange_{}.step".format(variant_id)
    filepath = os.path.join(out_dir, filename)
    shape.exportStep(filepath)
    print("  exported -> {}".format(filepath))


def main():
    out_dir = os.path.abspath(OUT_DIR)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    doe_path = os.path.abspath(DOE_CSV)
    if not os.path.isfile(doe_path):
        raise FileNotFoundError(
            "doe.csv not found at {}. Run flange_lhs_sampling.py first.".format(doe_path)
        )

    with open(doe_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    print("Loaded {} DOE rows from {}".format(len(rows), doe_path))

    for row in rows:
        variant_id = row["sample_id"]
        print("Building variant {} ...".format(variant_id))

        D   = float(row["D"])
        T   = float(row["T"])
        Db  = float(row["Db"])
        Hb  = float(row["Hb"])
        taper = float(row["taper"])
        dh  = float(row["dh"])
        boss_od = float(row["boss_od"])
        BCD = float(row["BCD"])
        R = BCD / 2.0
        fillet_large = float(row["fillet_large"])
        fillet_small = float(row["fillet_small"])

        doc, body, raw_shape = build_variant(row, variant_id)

        filleted_shape, n_applied, n_failed = fillet_flange_shape(
            raw_shape, D, T, Db, boss_od, Hb, taper, dh, R,
            fillet_large, fillet_small
        )
        print("    fillets applied: {}, skipped: {}".format(n_applied, n_failed))

        export_step(filleted_shape, variant_id, out_dir)
        App.closeDocument(doc.Name)

    print("Batch complete. STEP files saved in: {}".format(out_dir))


if __name__ == "__main__":
    main()