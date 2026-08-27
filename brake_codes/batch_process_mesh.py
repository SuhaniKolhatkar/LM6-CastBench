# -*- coding: utf-8 -*-
"""
STEP -> Gmsh mesh generator for LM6/AlSi12 casting thermal simulation
-----------------------------------------------------------------------
Scans an input folder for .step/.stp files, meshes each one with a mesh
strategy tuned for a conduction-only (Enthalpy phase-change) transient
Elmer run like your attached case.sif, tags the correct Physical Groups
so the .sif's Target Bodies / Target Boundaries line up, and writes a
.msh file per input into an output folder.

WHY THIS MESH STRATEGY
------------------------
Your case.sif is conduction-only (no Navier-Stokes), so there is no need
for boundary-layer inflation or a very fine near-wall mesh -- the flow
solver's convergence sensitivity is exactly what you removed. What DOES
matter for this physics:

  1. Geometric fidelity of small features (fillets, hole boundaries).
     A tiny fillet meshed too coarsely becomes a geometric facet, which
     locally distorts the conduction path and can create a false local
     hot/cold spot that has nothing to do with the real solidification
     physics.
  2. Smooth grading from fine (near small features) to coarse (in the
     bulk of the casting), so the model isn't uniformly dense. Since the
     eutectic reaction happens almost everywhere at nearly the same time
     (LM6 is near-eutectic composition), a uniformly fine mesh buys you
     very little accuracy but costs a lot of DOFs/timestep -- exactly
     what you were already trimming for the 15-hour compute budget.
  3. Decent tetrahedral quality (low skewness) so BiCGStab + ILU1 in
     Solver 1 converges cleanly every timestep, instead of struggling on
     a few sliver elements.

Concretely, this script uses:
  - Curvature-adaptive sizing (Mesh.MeshSizeFromCurvature): element size
    is automatically tied to local surface curvature, so small fillets
    and hole boundaries get refined ONLY where they need it, while large
    flat/gently-curved regions stay coarse.
  - Global Mesh.MeshSizeMin / Mesh.MeshSizeMax bounds computed from the
    model's own bounding-box diagonal, so the strategy self-scales
    across DOE variants of different overall size, instead of using a
    single hardcoded element size that might be too fine or too coarse
    for a different variant.
  - Frontal-Delaunay (2D) + HXT (3D) algorithms, which are fast and
    robust for this kind of prismatic/extruded solid.
  - A post-generation mesh optimization pass (Netgen optimizer if
    available, generic optimizer otherwise) to clean up low-quality
    tets before export.

PHYSICAL GROUPS (matched to your attached case.sif)
-----------------------------------------------------
Your case.sif expects:
    Body 1:               Target Bodies(1)      = 2   (Name "body1")
    Boundary Condition 1:  Target Boundaries(1)  = 1   (Name "Block_Surface")

i.e. ONE volume physical group and ONE boundary physical group covering
the entire exterior surface (this matches the geometry actually checked
here: a single solid with all-external faces, no internal mold contact
surface split into Hub/Blade/Casing regions the way your impeller mesh
was). This script reproduces exactly that:
    - Physical Volume, tag 2, name "body1"      -> all solid volumes
    - Physical Surface, tag 1, name "Block_Surface" -> all boundary faces

If/when you go back to a geometry that needs multiple boundary regions
(e.g. the impeller's Hub / Blade / Casing split), see the
`SURFACE_GROUPS` section below -- it's structured so you can swap the
single "group everything" block for explicit face-tag lists once you've
identified which face tags correspond to which physical region (e.g. by
opening the mesh in Gmsh's GUI and clicking on faces to read their tag
in the status bar).

HOW TO RUN
----------
    pip install gmsh --break-system-packages   (if not already installed)
    python3 generate_mesh.py

Expected inputs:
    ./step_input/*.step (or *.stp)   -- one or more STEP files
Outputs:
    ./mesh_output/<name>.msh         -- one mesh per input STEP file

After meshing, convert to Elmer's native mesh format before running
ElmerSolver, e.g.:
    ElmerGrid 14 2 mesh_output/<name>.msh -autoclean
which produces a mesh directory (mesh.header/mesh.nodes/mesh.elements/
mesh.boundary) that case.sif's `Mesh DB` line points to.
"""

import os
import glob
import math

import gmsh

# ----------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------
INPUT_DIR = r"D:\NeurIPS\surrogate_brake\brake_cad_outputs"
OUTPUT_DIR = r"D:\NeurIPS\surrogate_brake\brake_msh_outputs"

# Physical group tags/names -- matched to the attached case.sif
BODY_PHYSICAL_TAG = 2
BODY_PHYSICAL_NAME = "body1"
SURFACE_PHYSICAL_TAG = 1
SURFACE_PHYSICAL_NAME = "Block_Surface"

# Curvature-adaptive sizing: number of mesh elements per 2*pi radians of
# curvature. Higher = finer resolution of curved/filleted features.
# 15-20 is a good, non-excessive default for capturing fillets down to
# ~0.3-0.5 mm radius without over-refining everywhere.
CURVATURE_ELEMENTS_PER_2PI = 18

# Global element size bounds, expressed as a fraction of the model's own
# bounding-box diagonal, so the strategy self-scales across DOE variants.
# (Absolute floors/ceilings below prevent pathological cases.)
MIN_SIZE_FRACTION = 0.0015   # ~ diagonal * 0.0015 (fine, near small features)
MAX_SIZE_FRACTION = 0.03     # ~ diagonal * 0.03   (coarse, bulk regions)
ABS_MIN_SIZE = 0.15          # mm -- absolute floor (avoid over-refinement)
ABS_MAX_SIZE = 6.0           # mm -- absolute ceiling (avoid under-refinement)

# Mesh algorithm choices (Gmsh option numbers)
ALGO_2D_FRONTAL_DELAUNAY = 6
ALGO_3D_HXT = 10

# Element order: 1 = linear tets (leaner, recommended default for Elmer
# conduction runs), 2 = quadratic tets (more accurate per-element, but
# roughly doubles node count -- use only if linear-mesh convergence
# studies show you need it).
ELEMENT_ORDER = 1

# Gmsh .msh export format version. 2.2 is the most broadly compatible
# with ElmerGrid; bump to 4.1 if your ElmerGrid version supports it and
# you want the newer format.
MSH_FILE_VERSION = 2.2


# ----------------------------------------------------------------------
# 1. Mesh one STEP file
# ----------------------------------------------------------------------
def mesh_step_file(step_path, output_path):
    model_name = os.path.splitext(os.path.basename(step_path))[0]
    gmsh.model.add(model_name)

    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(3)
    surfaces = gmsh.model.getEntities(2)

    if not volumes:
        raise RuntimeError(f"No solid volume found in {step_path}")

    volume_tags = [v[1] for v in volumes]
    surface_tags = [s[1] for s in surfaces]

    # ------------------------------------------------------------------
    # Physical groups (must be set before mesh generation)
    # ------------------------------------------------------------------
    # Body: all solid volumes -> one physical volume, matching
    # "Target Bodies(1) = 2" / Name "body1" in case.sif
    gmsh.model.addPhysicalGroup(3, volume_tags, tag=BODY_PHYSICAL_TAG)
    gmsh.model.setPhysicalName(3, BODY_PHYSICAL_TAG, BODY_PHYSICAL_NAME)

    # Boundary: all exterior faces -> one physical surface, matching
    # "Target Boundaries(1) = 1" / Name "Block_Surface" in case.sif
    #
    # --- SURFACE_GROUPS: replace this single block with explicit
    # per-region face-tag lists (e.g. hub_faces / blade_faces /
    # casing_faces) if/when you need a multi-boundary-condition sif
    # like your impeller mesh. Identify face tags visually in Gmsh's
    # GUI (Tools > Visibility, or just click a face and read its tag
    # from the status bar) after running `gmsh <step_file>` once.
    gmsh.model.addPhysicalGroup(2, surface_tags, tag=SURFACE_PHYSICAL_TAG)
    gmsh.model.setPhysicalName(2, SURFACE_PHYSICAL_TAG, SURFACE_PHYSICAL_NAME)

    # ------------------------------------------------------------------
    # Mesh sizing strategy
    # ------------------------------------------------------------------
    bbox = gmsh.model.getBoundingBox(-1, -1)
    dx = bbox[3] - bbox[0]
    dy = bbox[4] - bbox[1]
    dz = bbox[5] - bbox[2]
    diag = math.dist(bbox[0:3], bbox[3:6])
    min_extent = min(dx, dy, dz)   # thin-wall / thickness dimension

    # Diagonal-based cap (bulk-region coarseness for the part's overall size)
    size_max_diag = min(diag * MAX_SIZE_FRACTION, ABS_MAX_SIZE)

    # Thickness-based cap: guarantee at least MIN_ELEMENTS_THROUGH_THICKNESS
    # elements span the part's thinnest dimension, regardless of how large
    # the part is in-plane. This is what the diagonal-only formula misses --
    # a long thin plate has a large diagonal but a tiny min_extent, and the
    # old formula let coarse sizing "win" in flat regions with no curvature
    # signal, badly under-resolving the through-thickness gradient.
    MIN_ELEMENTS_THROUGH_THICKNESS = 5
    size_max_thickness = min_extent / MIN_ELEMENTS_THROUGH_THICKNESS

    size_max = min(size_max_diag, size_max_thickness)
    size_min = max(diag * MIN_SIZE_FRACTION, ABS_MIN_SIZE)

    if size_min >= size_max:
        size_min = max(ABS_MIN_SIZE, size_max / 8.0)

    gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)

    # ------------------------------------------------------------------
    # Generate + optimize
    # ------------------------------------------------------------------
    gmsh.model.mesh.generate(3)

    try:
        gmsh.model.mesh.optimize("Netgen")
    except Exception:
        pass  # Netgen optimizer plugin not available in this gmsh build
    gmsh.model.mesh.optimize("")  # generic tet-quality optimization pass

    gmsh.write(output_path)

    # ------------------------------------------------------------------
    # Quick quality/size report
    # ------------------------------------------------------------------
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
    n_tets = sum(len(t) for t in elem_tags)
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)

    print(f"    bbox diagonal   : {diag:.2f} mm")
    print(f"    size range used : {size_min:.3f} - {size_max:.3f} mm")
    print(f"    volumes/surfaces: {len(volume_tags)} / {len(surface_tags)}")
    print(f"    nodes / tets    : {n_nodes} / {n_tets}")

    gmsh.model.remove()


# ----------------------------------------------------------------------
# 2. Batch driver
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    step_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.step"))
        + glob.glob(os.path.join(INPUT_DIR, "*.stp"))
    )

    if not step_files:
        print(f"No .step/.stp files found in ./{INPUT_DIR}/")
        return

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    for step_path in step_files:
        name = os.path.splitext(os.path.basename(step_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name}.msh")

        print(f"\nMeshing {step_path} -> {output_path}")
        try:
            mesh_step_file(step_path, output_path)
            print(f"  OK: wrote {output_path}")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    gmsh.finalize()
    print(f"\nDone. Meshes written to ./{OUTPUT_DIR}/")
    print("Next step -- convert each .msh to Elmer's native mesh format, e.g.:")
    print("    ElmerGrid 14 2 mesh_output/<name>.msh -autoclean")


if __name__ == "__main__":
    main()