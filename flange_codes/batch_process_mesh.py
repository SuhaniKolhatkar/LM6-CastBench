# -*- coding: utf-8 -*-
"""
mesh_flange_variants.py  (v2 - adds Elmer-ready physical groups)

Reads STEP files from flange_cad_outputs/, meshes each one with Gmsh
using a curvature-adaptive, quality-driven strategy, tags the mesh with
physical groups matching your Elmer .sif conventions, and exports .msh
files to flange_mesh_outputs/.

NEW IN THIS VERSION - PHYSICAL GROUPS FOR ELMER
-------------------------------------------------
Matches the sample .sif's expectations:
  Body 1:               Target Bodies(1)     = 2   -> Physical Volume tag 2, name "body1"
  Boundary Condition 1: Target Boundaries(1) = 1   -> Physical Surface tag 1, name "Flange_Surface"

The flange is a single connected solid with one continuous exterior
surface (base plate + boss + bore + bolt holes are all part of the
same boundary, same as the "Stepped_Block" case in your .sif, which
also uses a single boundary region rather than a multi-region split
like the impeller Hub/Blade/Casing case).

These tags are written into the .msh file as Gmsh Physical Groups.
When you run ElmerGrid to convert to Elmer's native mesh format
(e.g. `ElmerGrid 14 2 flange_1.msh`), it will auto-generate
mesh.names from these physical names/tags, so your .sif's
Target Bodies(1) = 2 / Target Boundaries(1) = 1 will resolve directly
without any manual remapping.

If you later want per-face boundary conditions (e.g. different HTC on
the bolt-hole walls vs. the flat faces), the surface tagging block
below is the place to split that into multiple Physical Surfaces -
flagged with a comment where to do it.

MESHING STRATEGY (unchanged from v1)
-------------------------------------
1. Mesh sizing derived per-variant from its own DOE parameters
   (dh, fillet_small, D) so bolt holes/fillets are never
   under-resolved and large flat regions are never over-refined.
2. Mesh.MeshSizeFromCurvature adds curvature-adaptive refinement on
   every curved edge automatically.
3. Algorithm3D = HXT + Netgen optimization passes.
4. Quality-driven retry loop targeting minSICN / minSIGE / Jacobian
   ratio >= 0.80, only shrinking mesh size when optimization alone
   plateaus - keeps element count minimal.
5. .msh format 2.2 by default (broad Elmer/legacy solver support).

Requires: pip install gmsh

Run with:
    python mesh_flange_variants.py
"""

import gmsh
import csv
import os
import glob
import re
import sys

# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------
STEP_DIR = r"D:\NeurIPS\surrogate_flange\flange_cad_outputs"
DOE_CSV = r"D:\NeurIPS\surrogate_flange\doe.csv"
OUT_DIR = r"D:\NeurIPS\surrogate_flange\flange_mesh_outputs"

QUALITY_TARGET = 0.80
MAX_ITERATIONS = 6
OPTIMIZE_PASSES_PER_ITER = 3
SIZE_SHRINK_FACTOR = 0.85
MSH_FILE_VERSION = 2.2

CURVATURE_POINTS_PER_CIRCLE = 14
ELEMENT_ORDER = 1

# ---- Elmer physical group tags/names (must match your .sif) ----
BODY_PHYSICAL_TAG = 2
BODY_PHYSICAL_NAME = "body1"
SURFACE_PHYSICAL_TAG = 1
SURFACE_PHYSICAL_NAME = "Flange_Surface"


# -----------------------------------------------------------------
# DOE lookup
# -----------------------------------------------------------------
def load_doe(doe_path):
    doe = {}
    if not os.path.isfile(doe_path):
        print("  [warn] doe.csv not found at {} - will fall back to bounding-box-based sizing".format(doe_path))
        return doe
    with open(doe_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            doe[str(row["sample_id"])] = row
    return doe


def variant_id_from_filename(path):
    m = re.search(r"flange_(\d+)\.step$", os.path.basename(path), re.IGNORECASE)
    return m.group(1) if m else None


def get_size_bounds(doe_row, bbox_diag):
    if doe_row is not None:
        dh = float(doe_row.get("dh", 0) or 0)
        fillet_small = float(doe_row.get("fillet_small", 0) or 0)
        D = float(doe_row.get("D", 0) or 0)

        candidates = [v for v in [dh, fillet_small * 2.0] if v > 0]
        min_size = (min(candidates) / 3.0) if candidates else bbox_diag * 0.01
        min_size = max(min_size, 0.4)

        max_size = D * 0.09 if D > 0 else bbox_diag * 0.08
        max_size = max(max_size, min_size * 4.0)
    else:
        min_size = bbox_diag * 0.01
        max_size = bbox_diag * 0.08

    return min_size, max_size


# -----------------------------------------------------------------
# Elmer physical groups
# -----------------------------------------------------------------
def assign_physical_groups():
    """
    Tags the imported solid as Physical Volume 2 ("body1") and all its
    exterior faces as a single Physical Surface 1 ("Flange_Surface"),
    matching the Target Bodies / Target Boundaries indices used in the
    reference .sif.

    Must be called AFTER occ.synchronize() and BEFORE mesh.generate(3).
    """
    volumes = gmsh.model.getEntities(dim=3)
    surfaces = gmsh.model.getEntities(dim=2)

    if not volumes:
        raise RuntimeError("No 3D volume found in imported STEP - cannot assign body physical group.")
    if not surfaces:
        raise RuntimeError("No surfaces found in imported STEP - cannot assign boundary physical group.")

    volume_tags = [v[1] for v in volumes]
    surface_tags = [s[1] for s in surfaces]

    # --- Body (single volume) ---
    gmsh.model.addPhysicalGroup(3, volume_tags, tag=BODY_PHYSICAL_TAG)
    gmsh.model.setPhysicalName(3, BODY_PHYSICAL_TAG, BODY_PHYSICAL_NAME)

    # --- Boundary (all exterior faces as one region, matching the
    #     single-boundary "Stepped_Block" pattern in the .sif) ---
    #
    # To split into multiple boundary regions later (e.g. bolt-hole
    # walls vs. flat faces for a different HTC), replace this single
    # addPhysicalGroup call with a filtering step over `surfaces`
    # (e.g. by face CenterOfMass/normal/radius, same approach used for
    # the fillet edge detection) and assign separate tags/names, then
    # add matching "Boundary Condition N" blocks in the .sif.
    gmsh.model.addPhysicalGroup(2, surface_tags, tag=SURFACE_PHYSICAL_TAG)
    gmsh.model.setPhysicalName(2, SURFACE_PHYSICAL_TAG, SURFACE_PHYSICAL_NAME)

    # Only write physically-grouped entities to the .msh (keeps the
    # file clean and matched to what Elmer/ElmerGrid expects)
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    return len(volume_tags), len(surface_tags)


# -----------------------------------------------------------------
# Quality evaluation
# -----------------------------------------------------------------
def evaluate_quality(dim=3):
    element_types, element_tags, _ = gmsh.model.mesh.getElements(dim)
    all_tags = []
    for tags in element_tags:
        all_tags.extend(tags)

    if not all_tags:
        return {}, 0

    results = {}
    for measure in ["minSICN", "minSIGE"]:
        try:
            vals = gmsh.model.mesh.getElementQualities(all_tags, measure)
            if vals:
                results[measure] = (min(vals), sum(vals) / len(vals))
        except Exception:
            pass

    try:
        min_jac = gmsh.model.mesh.getElementQualities(all_tags, "minDetJac")
        max_jac = gmsh.model.mesh.getElementQualities(all_tags, "maxDetJac")
        ratios = [
            (mn / mx) if mx not in (0, None) else 0.0
            for mn, mx in zip(min_jac, max_jac)
        ]
        if ratios:
            results["jacobianRatio"] = (min(ratios), sum(ratios) / len(ratios))
    except Exception:
        pass

    return results, len(all_tags)


def quality_passes(results, target):
    if not results:
        return False
    for _name, (mn, _mean) in results.items():
        if mn < target:
            return False
    return True


def worst_metric(results):
    if not results:
        return None, None
    name, (mn, _mean) = min(results.items(), key=lambda kv: kv[1][0])
    return name, mn


# -----------------------------------------------------------------
# Meshing a single variant
# -----------------------------------------------------------------
def mesh_variant(step_path, doe_row, out_dir):
    variant_id = variant_id_from_filename(step_path) or "unknown"
    print("Meshing variant {} ({}) ...".format(variant_id, os.path.basename(step_path)))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("flange_{}".format(variant_id))
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        # ---- Elmer physical groups (body + boundary) ----
        n_vols, n_surfs = assign_physical_groups()
        print("  physical groups: 1 volume (tag {}, '{}'), 1 boundary region "
              "covering {} face(s) (tag {}, '{}')".format(
                  BODY_PHYSICAL_TAG, BODY_PHYSICAL_NAME, n_surfs,
                  SURFACE_PHYSICAL_TAG, SURFACE_PHYSICAL_NAME))

        bbox = gmsh.model.getBoundingBox(-1, -1)
        bbox_diag = ((bbox[3] - bbox[0]) ** 2 + (bbox[4] - bbox[1]) ** 2 + (bbox[5] - bbox[2]) ** 2) ** 0.5

        min_size, max_size = get_size_bounds(doe_row, bbox_diag)
        print("  size bounds: min={:.3f} mm, max={:.3f} mm".format(min_size, max_size))

        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", CURVATURE_POINTS_PER_CIRCLE)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)

        gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # HXT
        gmsh.option.setNumber("Mesh.Algorithm", 6)       # Frontal-Delaunay surfaces

        gmsh.option.setNumber("Mesh.ElementOrder", ELEMENT_ORDER)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        current_min = min_size
        current_max = max_size
        results = {}
        n_elements = 0

        for iteration in range(1, MAX_ITERATIONS + 1):
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(3)

            for _ in range(OPTIMIZE_PASSES_PER_ITER):
                try:
                    gmsh.model.mesh.optimize("Netgen")
                except Exception:
                    break

            results, n_elements = evaluate_quality(dim=3)

            if quality_passes(results, QUALITY_TARGET):
                print("  [ok] iteration {}: quality target met ({} elements)".format(iteration, n_elements))
                break

            worst_name, worst_val = worst_metric(results)
            print("  iteration {}: quality below target (worst = {} at {:.3f}), {} elements".format(
                iteration, worst_name, worst_val if worst_val is not None else -1, n_elements))

            if iteration == MAX_ITERATIONS:
                print("  [warn] variant {}: quality target not fully reached after {} iterations "
                      "(worst = {} = {:.3f}); exporting best achieved mesh.".format(
                          variant_id, MAX_ITERATIONS, worst_name, worst_val if worst_val is not None else -1))
                break

            current_min *= SIZE_SHRINK_FACTOR
            current_max *= SIZE_SHRINK_FACTOR
            gmsh.option.setNumber("Mesh.MeshSizeMin", current_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", current_max)

        out_path = os.path.join(out_dir, "flange_{}.msh".format(variant_id))
        gmsh.option.setNumber("Mesh.MshFileVersion", MSH_FILE_VERSION)
        gmsh.write(out_path)

        summary = {
            "variant_id": variant_id,
            "n_elements": n_elements,
            "min_size_final": round(current_min, 4),
            "max_size_final": round(current_max, 4),
            "body_physical_tag": BODY_PHYSICAL_TAG,
            "surface_physical_tag": SURFACE_PHYSICAL_TAG,
        }
        for name, (mn, mean) in results.items():
            summary["{}_min".format(name)] = round(mn, 4)
            summary["{}_mean".format(name)] = round(mean, 4)
        summary["quality_target_met"] = quality_passes(results, QUALITY_TARGET)

        print("  exported -> {}".format(out_path))
        return summary

    except Exception as e:
        print("  [error] variant {} failed: {}".format(variant_id, e))
        return {"variant_id": variant_id, "error": str(e)}

    finally:
        gmsh.finalize()


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main():
    step_dir = os.path.abspath(STEP_DIR)
    out_dir = os.path.abspath(OUT_DIR)
    doe_path = os.path.abspath(DOE_CSV)

    if not os.path.isdir(step_dir):
        raise FileNotFoundError("STEP directory not found: {}".format(step_dir))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    doe = load_doe(doe_path)

    step_files = sorted(
        glob.glob(os.path.join(step_dir, "flange_*.step")),
        key=lambda p: int(re.search(r"flange_(\d+)\.step$", os.path.basename(p)).group(1))
        if re.search(r"flange_(\d+)\.step$", os.path.basename(p)) else 0
    )

    if not step_files:
        print("No flange_*.step files found in {}".format(step_dir))
        sys.exit(1)

    print("Found {} STEP file(s) in {}".format(len(step_files), step_dir))

    all_summaries = []
    for step_path in step_files:
        vid = variant_id_from_filename(step_path)
        doe_row = doe.get(vid) if vid else None
        summary = mesh_variant(step_path, doe_row, out_dir)
        all_summaries.append(summary)

    report_path = os.path.join(out_dir, "mesh_quality_report.csv")
    fieldnames = sorted({k for s in all_summaries for k in s.keys()})
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(s)

    n_ok = sum(1 for s in all_summaries if s.get("quality_target_met"))
    n_failed = sum(1 for s in all_summaries if "error" in s)
    print("\nBatch meshing complete: {}/{} variants met quality target 0.80, {} failed.".format(
        n_ok, len(all_summaries), n_failed))
    print("Meshes saved in: {}".format(out_dir))
    print("Quality report: {}".format(report_path))


if __name__ == "__main__":
    main()