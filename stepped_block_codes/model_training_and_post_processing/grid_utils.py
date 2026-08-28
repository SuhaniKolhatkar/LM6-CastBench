"""
grid_utils.py

WHAT THIS DOES
---------------
U-Net needs a REGULAR VOXEL GRID (fixed-size 3D tensor), but your mesh
data is an UNSTRUCTURED point cloud (variable node count per variant).
This file converts between the two:

  MESH -> GRID (regridding): for each variant, build a bounding box
  around its nodes, lay down a GRID_RES^3 voxel grid, and for every
  voxel find its NEAREST mesh node (via a KD-tree). That nearest
  node's SDF / temperature value is assigned to the voxel. Voxels
  farther than a distance threshold from any real node are marked
  "empty" (occupancy=0) -- these are outside the actual casting
  geometry (e.g. air/outside-the-part regions in the bounding box).

  GRID -> MESH (un-projecting predictions): after the U-Net predicts
  a full grid, we need per-NODE temperature values to compute your
  C1-C6 metrics on the same basis as MLP/GINO (fair comparison). This
  is a fast, DETERMINISTIC index lookup (no second KD-tree needed):
  each node's exact voxel is computed directly from its position and
  the same bounding box / voxel size used to build the grid.

IMPORTANT LIMITATION (worth reporting in your paper): regridding mesh
data onto a fixed grid and back introduces a small "quantization
error" even before any learning happens -- multiple mesh nodes that
fall in the same voxel all get identical predicted values. This is a
real, inherent trade-off of the U-Net/grid approach versus MLP/GINO,
which operate on exact node positions.

The NEAREST-NODE mapping (nearest_idx, valid_mask, node_voxel_idx) is
computed ONCE per variant, using nearest-neighbor / voxel-index
lookups (not recomputed every epoch -- the mesh geometry doesn't
change across epochs, only the temperature values being displayed
change per timestep).

Requires: pip install scipy numpy torch
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


def build_grid_mapping(pos, grid_res, padding_frac=0.05):
    """
    Computes, ONCE per variant, everything needed to regrid that
    variant's mesh onto a GRID_RES^3 voxel grid, and to un-project
    grid predictions back onto the original mesh nodes.

    pos: [N, 3] numpy array of node positions (meters)
    grid_res: int, voxel grid resolution per axis (e.g. 32 -> 32^3 grid)

    Returns a dict:
      nearest_idx    : [grid_res^3] int array, nearest mesh node per voxel
      valid_mask     : [grid_res^3] float array, 1.0 if voxel is "inside"
                        the geometry (close to a real node), else 0.0
      node_voxel_idx : [N] int array (flattened), each node's voxel index
                        -- used to un-project grid predictions -> nodes
      bbox_min, voxel_size : geometry info, kept for reference/debugging
    """
    bbox_min = pos.min(axis=0) - padding_frac * (pos.max(axis=0) - pos.min(axis=0) + 1e-6)
    bbox_max = pos.max(axis=0) + padding_frac * (pos.max(axis=0) - pos.min(axis=0) + 1e-6)
    voxel_size = (bbox_max - bbox_min) / grid_res

    lin = [np.linspace(bbox_min[d] + voxel_size[d] / 2,
                        bbox_max[d] - voxel_size[d] / 2, grid_res) for d in range(3)]
    gx, gy, gz = np.meshgrid(*lin, indexing="ij")
    grid_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    tree = cKDTree(pos)
    dist, nearest_idx = tree.query(grid_points, k=1)

    voxel_diag = np.linalg.norm(voxel_size)
    threshold = voxel_diag * 1.5
    valid_mask = (dist < threshold).astype(np.float32)

    # Deterministic reverse mapping: node position -> voxel index
    node_voxel_idx_3d = np.floor((pos - bbox_min) / voxel_size).astype(int)
    node_voxel_idx_3d = np.clip(node_voxel_idx_3d, 0, grid_res - 1)
    node_voxel_idx = (node_voxel_idx_3d[:, 0] * grid_res * grid_res +
                       node_voxel_idx_3d[:, 1] * grid_res +
                       node_voxel_idx_3d[:, 2])

    return {
        "nearest_idx": nearest_idx,
        "valid_mask": valid_mask,
        "node_voxel_idx": node_voxel_idx,
        "bbox_min": bbox_min,
        "voxel_size": voxel_size,
        "grid_res": grid_res,
    }


def build_input_grid(mapping, sdf, pour_temp, mould_temp, time_val, grid_res):
    """
    Builds the U-Net's INPUT tensor for one (variant, timestep):
    5 channels stacked as [C, D, H, W]:
      0: occupancy mask (1.0 inside geometry, 0.0 outside)
      1: SDF (regridded from nearest node, 0 outside geometry)
      2: pour_temp (constant value broadcast across the whole grid)
      3: mould_temp (constant value broadcast across the whole grid)
      4: time (constant value broadcast across the whole grid)
    """
    nearest_idx = mapping["nearest_idx"]
    valid_mask = mapping["valid_mask"]

    occupancy = valid_mask.reshape(grid_res, grid_res, grid_res)
    sdf_grid = np.where(valid_mask > 0, sdf[nearest_idx], 0.0).reshape(grid_res, grid_res, grid_res)

    pour_grid = np.full((grid_res, grid_res, grid_res), pour_temp, dtype=np.float32)
    mould_grid = np.full((grid_res, grid_res, grid_res), mould_temp, dtype=np.float32)
    time_grid = np.full((grid_res, grid_res, grid_res), time_val, dtype=np.float32)

    stacked = np.stack([occupancy, sdf_grid, pour_grid, mould_grid, time_grid], axis=0)
    return torch.from_numpy(stacked.astype(np.float32))


def build_target_grid(mapping, temp_at_t, grid_res):
    """Regrids the true temperature field at one timestep onto the grid (1 channel)."""
    nearest_idx = mapping["nearest_idx"]
    valid_mask = mapping["valid_mask"]
    temp_grid = np.where(valid_mask > 0, temp_at_t[nearest_idx], 0.0).reshape(grid_res, grid_res, grid_res)
    return torch.from_numpy(temp_grid[None, ...].astype(np.float32))  # [1, D, H, W]


def unproject_grid_to_nodes(pred_grid, mapping):
    """
    Takes a predicted grid [D, H, W] (already de-normalized, real Kelvin
    values) and returns per-NODE predictions [N] using the deterministic
    node_voxel_idx computed once in build_grid_mapping -- no KD-tree
    needed here, just an index lookup.
    """
    flat_pred = pred_grid.reshape(-1)
    node_voxel_idx = mapping["node_voxel_idx"]
    return flat_pred[node_voxel_idx]