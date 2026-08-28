"""
Compute eutectic-plateau (850-886K) MAE / relative L2 / P95 metrics for
MLP and U-Net on ALL 15 Flange test-set variants -- for Table 7.

Same script as Brake's version, repointed at Flange's files/paths.
Requires:
  MLP:   mlp_flange.pt, norm_stats_flange.pt   <-- NOTE: verify this
         is the SAME norm_stats file used for your official
         metrics_flange.csv (the checkpoint_flange.pt / mlp_flange.pt
         mix-up earlier means you should double check which model file
         is actually correct for Flange -- checkpoint_flange.pt was
         confirmed as the one matching metrics_flange.csv, NOT mlp_flange.pt.
         See MLP_MODEL_PATH below -- set to the confirmed-correct file.
  U-Net: unet_flange.pt, norm_stats_unet_flange.pt, grid_utils.py, unet_model.py
  Both:  flange_splits/test.pt
"""

import os
import numpy as np
import pandas as pd
import torch

from grid_utils import build_grid_mapping, build_input_grid, unproject_grid_to_nodes
from unet_model import TemperatureUNet3D

# ---- EDIT THESE ----
SPLITS_DIR           = r"E:\NeurIPS_dataset\flange_models\flange_splits"

# IMPORTANT: for Flange, the file that matched metrics_flange.csv was
# checkpoint_flange.pt (wrapped dict, needs ["model_state"]), NOT
# mlp_flange.pt. Confirm this is still correct before running.
MLP_MODEL_PATH       = r"E:\NeurIPS_dataset\flange_models\flange_results\mlp\checkpoint_flange.pt"
MLP_IS_WRAPPED_CKPT  = True   # True = checkpoint dict with ["model_state"], False = raw state_dict
MLP_NORM_STATS_PATH  = r"E:\NeurIPS_dataset\flange_models\flange_results\mlp\norm_stats_flange.pt"

UNET_MODEL_PATH      = r"E:\NeurIPS_dataset\flange_models\flange_results\unet\unet_flange.pt"
UNET_NORM_STATS_PATH = r"E:\NeurIPS_dataset\flange_models\flange_results\unet\norm_stats_unet_flange.pt"
GRID_RES             = 32
EUTECTIC_LOW, EUTECTIC_HIGH = 850.0, 886.0
OUT_CSV              = r"E:\NeurIPS_dataset\flange_models\eutectic_metrics_flange.csv"
DEVICE = torch.device("cpu")   # forced CPU -- avoids the CUDA/CPU stats mismatch seen earlier
# ---------------------


def load_split(name):
    return torch.load(os.path.join(SPLITS_DIR, f"{name}.pt"), weights_only=False)


class MLP(torch.nn.Module):
    def __init__(self, in_dim=7, out_dim=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def eutectic_metrics(gt, pred, low=EUTECTIC_LOW, high=EUTECTIC_HIGH):
    """Restrict to nodes/timesteps where the TRUE temperature is in [low, high]."""
    mask = (gt >= low) & (gt <= high)
    n_pts = mask.sum()
    if n_pts == 0:
        return None, None, None, 0
    err = pred[mask] - gt[mask]
    abs_err = np.abs(err)
    mae = abs_err.mean()
    rel_l2 = np.linalg.norm(err) / np.linalg.norm(gt[mask]).clip(min=1e-8)
    p95 = np.percentile(abs_err, 95)
    return float(mae), float(rel_l2), float(p95), int(n_pts)


def run_mlp(test_list):
    model = MLP()
    raw = torch.load(MLP_MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = raw["model_state"] if MLP_IS_WRAPPED_CKPT else raw
    model.load_state_dict(state_dict)
    model.eval()

    stats = torch.load(MLP_NORM_STATS_PATH, map_location="cpu", weights_only=False)
    in_mean, in_std = stats["input_mean"], stats["input_std"]
    out_mean, out_std = stats["target_mean"], stats["target_std"]

    rows = []
    for data in test_list:
        n_nodes, n_steps = data.x.shape[0], data.y.shape[1]
        static_rep = data.x.unsqueeze(1).expand(n_nodes, n_steps, 6)
        time_rep = data.time_stamps.view(1, n_steps, 1).expand(n_nodes, n_steps, 1)
        inputs = torch.cat([static_rep, time_rep], dim=2).reshape(n_nodes * n_steps, 7)
        inputs_n = (inputs - in_mean) / in_std
        with torch.no_grad():
            pred_n = model(inputs_n)
        pred = (pred_n * out_std + out_mean).reshape(n_nodes, n_steps).numpy()
        gt = data.y.numpy()

        mae, rel_l2, p95, n_pts = eutectic_metrics(gt, pred)
        rows.append({"model": "MLP", "sim_id": data.sim_id,
                      "PC_MAE_K": mae, "PC_rel_L2": rel_l2, "PC_P95_K": p95,
                      "n_eutectic_points": n_pts})
        print(f"MLP  {data.sim_id}: PC_MAE={mae}  PC_relL2={rel_l2}  PC_P95={p95}  n_pts={n_pts}")
    return rows


def run_unet(test_list):
    model = TemperatureUNet3D().to(DEVICE)
    model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    stats = torch.load(UNET_NORM_STATS_PATH, map_location="cpu", weights_only=False)

    rows = []
    for data in test_list:
        pos_np = data.pos.numpy()
        sdf_np = data.sdf.numpy()
        n_nodes, n_steps = data.pos.shape[0], data.y.shape[1]
        mapping = build_grid_mapping(pos_np, GRID_RES)

        pred_full = torch.zeros(n_nodes, n_steps)
        with torch.no_grad():
            for t in range(n_steps):
                time_val = float(data.time_stamps[t])
                x_grid = build_input_grid(mapping, sdf_np, float(data.pour_temp),
                                           float(data.mould_temp), time_val, GRID_RES)
                x_n = (x_grid - stats["input_mean"].squeeze(0)) / stats["input_std"].squeeze(0)
                x_n = x_n.unsqueeze(0).to(DEVICE)
                pred_grid_n = model(x_n).squeeze(0).squeeze(0).cpu()
                pred_grid = pred_grid_n * stats["target_std"].squeeze() + stats["target_mean"].squeeze()
                pred_nodes = unproject_grid_to_nodes(pred_grid.numpy(), mapping)
                pred_full[:, t] = torch.from_numpy(pred_nodes)

        gt = data.y.numpy()
        pred = pred_full.numpy()
        mae, rel_l2, p95, n_pts = eutectic_metrics(gt, pred)
        rows.append({"model": "U-Net", "sim_id": data.sim_id,
                      "PC_MAE_K": mae, "PC_rel_L2": rel_l2, "PC_P95_K": p95,
                      "n_eutectic_points": n_pts})
        print(f"U-Net  {data.sim_id}: PC_MAE={mae}  PC_relL2={rel_l2}  PC_P95={p95}  n_pts={n_pts}")
    return rows


def main():
    test_list = load_split("test")
    print(f"Loaded {len(test_list)} test-set variants\n")

    mlp_rows = run_mlp(test_list)
    unet_rows = run_unet(test_list)

    df = pd.DataFrame(mlp_rows + unet_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved per-sim eutectic metrics -> {OUT_CSV}")

    print("\n=== Table 7 row values (mean across test set, excluding sims with 0 eutectic points) ===")
    for model in ["MLP", "U-Net"]:
        sub = df[(df["model"] == model) & (df["n_eutectic_points"] > 0)]
        n_total = len(df[df["model"] == model])
        if len(sub) == 0:
            print(f"{model}: no test sims reached the eutectic range -- check EUTECTIC_LOW/HIGH or data")
            continue
        print(f"{model:6s}  PC-MAE={sub['PC_MAE_K'].mean():.3f}K  "
              f"PC-relL2={sub['PC_rel_L2'].mean():.4f}  PC-P95={sub['PC_P95_K'].mean():.3f}K  "
              f"(coverage: {len(sub)}/{n_total} sims reached plateau)")


if __name__ == "__main__":
    main()