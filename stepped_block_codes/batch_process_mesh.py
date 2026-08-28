import gmsh
import os
import glob

# ---------------------------------------------------------------------
# CONFIG - edit these two paths
# ---------------------------------------------------------------------
STEP_DIR = r"D:\NeurIPS\surrogate_block\block_step_outputs"
MSH_DIR  = r"D:\NeurIPS\surrogate_block\block_mesh_outputs"

MESH_SIZE_MIN = 2.0   # mm
MESH_SIZE_MAX = 5.0   # mm

os.makedirs(MSH_DIR, exist_ok=True)


def mesh_step_file(step_path, msh_path):
    """Import one STEP file, group surfaces/volume, mesh it, and write .msh."""
    model_name = os.path.splitext(os.path.basename(step_path))[0]

    gmsh.model.add(model_name)
    gmsh.model.setCurrent(model_name)

    # ── Import the STEP geometry ─────────────────────────────────────
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    # ── Group ALL surfaces into one Physical Surface ─────────────────
    surfaces = gmsh.model.getEntities(dim=2)
    surface_tags = [tag for (dim, tag) in surfaces]
    print(f"  Found {len(surface_tags)} surfaces: {surface_tags}")
    gmsh.model.addPhysicalGroup(2, surface_tags, name="Block_Surface")

    # ── Group the (single) volume into one Physical Volume ───────────
    volumes = gmsh.model.getEntities(dim=3)
    volume_tags = [tag for (dim, tag) in volumes]
    print(f"  Found {len(volume_tags)} volume(s): {volume_tags}")
    gmsh.model.addPhysicalGroup(3, volume_tags, name="Block")

    # ── Mesh settings ──────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MIN)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX)

    # ── Generate 3D tetrahedral mesh ───────────────────────────────────
    gmsh.model.mesh.generate(3)

    # ── Write out in Gmsh format 2.2 ASCII — what ElmerGrid 14 2 expects
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(msh_path)

    # Clear this model out of memory before moving to the next file
    gmsh.model.remove()


def main():
    step_files = sorted(glob.glob(os.path.join(STEP_DIR, "*.step")))
    print(f"Found {len(step_files)} STEP files in {STEP_DIR}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    failed = []
    for i, step_path in enumerate(step_files, start=1):
        base_name = os.path.splitext(os.path.basename(step_path))[0]
        msh_path = os.path.join(MSH_DIR, base_name + ".msh")

        print(f"[{i}/{len(step_files)}] Meshing {os.path.basename(step_path)} ...")
        try:
            mesh_step_file(step_path, msh_path)
            print(f"  -> saved {msh_path}")
        except Exception as e:
            print(f"  !! FAILED: {e}")
            failed.append(step_path)

    gmsh.finalize()

    print("\nDone.")
    print(f"  Succeeded: {len(step_files) - len(failed)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print("  Failed files:")
        for f in failed:
            print(f"   - {f}")
    print(f"\nRun ElmerGrid on each output, e.g.: ElmerGrid 14 2 {MSH_DIR}/<name>.msh")


if __name__ == "__main__":
    main()