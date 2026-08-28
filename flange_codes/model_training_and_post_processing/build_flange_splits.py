"""
build_brake_splits.py

WHAT THIS DOES
---------------
Builds an 80/10/10 train/val/test split for a SINGLE geometry family
(Brake), reading directly from the raw .pt dict files in
brake_data_pt/ (no dependency on the old master_dataset.csv path,
since the data has been moved to a new per-geometry folder).

Each raw .pt file is a plain dict with keys:
    pos, edge_index, temp_history, sdf, node_features,
    pour_temp, mould_temp, n_timesteps, timestep_indices,
    time_stamps, dt

This script wraps each dict into a PyTorch Geometric Data object
(x=node_features, edge_index=edge_index, y=temp_history) and saves
three lists: train.pt, val.pt, test.pt -- ready to be loaded directly
by any of the three model training scripts (MLP / GINO / U-Net).

WHY A SEPARATE SPLIT PER GEOMETRY
-----------------------------------
You are training one surrogate PER geometry (Brake first), not a
pooled multi-geometry model this time -- so the split is scoped
entirely to brake_data_pt, unlike the earlier pooled-splits script.

Requires: pip install torch torch_geometric scikit-learn
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# CONFIG -- change these three lines when you move to Flange / Block
# ---------------------------------------------------------------------
GEOMETRY_FAMILY = "Brake"
RAW_PT_DIR = r"E:\NeurIPS_dataset\flange_models\flange_data_pt"
SPLITS_DIR = r"E:\NeurIPS_dataset\flange_models\flange_splits"

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
RANDOM_SEED = 42

os.makedirs(SPLITS_DIR, exist_ok=True)


def load_as_pyg_data(pt_path):
    raw = torch.load(pt_path, weights_only=False)
    data = Data(
        x=raw["node_features"],        # [N, 6]
        edge_index=raw["edge_index"],  # [2, E]
        y=raw["temp_history"],         # [N, T]
        pos=raw["pos"],                # [N, 3] meters
    )
    data.sdf = raw["sdf"]
    data.timestep_indices = raw["timestep_indices"]
    data.time_stamps = raw["time_stamps"]
    data.dt = raw["dt"]
    data.pour_temp = raw["pour_temp"]
    data.mould_temp = raw["mould_temp"]
    data.n_timesteps = raw["n_timesteps"]
    data.geometry_family = GEOMETRY_FAMILY
    data.sim_id = os.path.splitext(os.path.basename(pt_path))[0]
    return data


def build_splits():
    pt_files = sorted(glob.glob(os.path.join(RAW_PT_DIR, "*.pt")))
    print(f"Found {len(pt_files)} .pt files in {RAW_PT_DIR}")
    if len(pt_files) < 3:
        raise RuntimeError("Too few .pt files to split 80/10/10.")

    train_files, remaining = train_test_split(
        pt_files, train_size=TRAIN_FRAC, random_state=RANDOM_SEED
    )
    val_frac_of_remaining = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_files, test_files = train_test_split(
        remaining, train_size=val_frac_of_remaining, random_state=RANDOM_SEED
    )

    print(f"train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        data_list = []
        for f in files:
            try:
                data_list.append(load_as_pyg_data(f))
            except Exception as e:
                print(f"FAIL loading {f}: {e}")
        out_path = os.path.join(SPLITS_DIR, f"{split_name}.pt")
        torch.save(data_list, out_path)
        print(f"Saved {split_name}: {len(data_list)} graphs -> {out_path}")


if __name__ == "__main__":
    build_splits()