"""
STAGE 1 (UNet): Run the trained U-Net on every test-set variant for
Stepped Block, save per-node predictions/ground truth in the shared
.npz format, and build a summary CSV for cross-checking against
metrics_unet_stepped_block.csv.

Only runs on the TEST split (test.pt).
"""

import os
import time
import numpy as np
import pandas as pd
import torch

from grid_utils import build_grid_mapping, build_input_grid, unproject_grid_to_nodes
from unet_model import TemperatureUNet3D

# ---- EDIT THESE ----
GEOMETRY_NAME    = "stepped_block"
SPLITS_DIR       = r"E:\NeurIPS_dataset\stepped_block\stepped_block_splits"   # must contain test.pt
MODEL_PATH       = r"E:\NeurIPS_dataset\stepped_block\stepped_block_results\unet\unet_stepped_block.pt"
NORM_STATS_PATH  = r"E:\NeurIPS_dataset\stepped_block\stepped_block_results\unet\norm_stats_unet_stepped_block.pt"
GRID_RES         = 32
OUT_DIR          = r"E:\NeurIPS_dataset\stepped_block\field_data_stepped_block_UNET"
DEVICE           = torch.device("cpu")   # avoids the CPU/CUDA stats mismatch from Brake's run
# ---------------------

os.makedirs(OUT_DIR, exist_ok=True)


def load_split(name):
    return torch.load(os.path.join(SPLITS_DIR, f"{name}.pt"), weights_only=False)


def main():
    model = TemperatureUNet3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    stats = torch.load(NORM_STATS_PATH, map_location="cpu", weights_only=False)

    test_list = load_split("test")
    print(f"Loaded {len(test_list)} test-set variants for {GEOMETRY_NAME}")

    summary_rows = []
    for data in test_list:
        pos_np = data.pos.numpy()
        sdf_np = data.sdf.numpy()
        n_nodes = data.pos.shape[0]
        n_steps = data.y.shape[1]

        mapping = build_grid_mapping(pos_np, GRID_RES)

        true_full = data.y  # [N, T]
        pred_full = torch.zeros(n_nodes, n_steps)

        t0 = time.perf_counter()
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
        infer_time = time.perf_counter() - t0

        gt_temp = true_full.numpy()
        pred_temp = pred_full.numpy()
        abs_err = np.abs(pred_temp - gt_temp)

        out_path = os.path.join(OUT_DIR, f"field_data_{data.sim_id}.npz")
        np.savez(
            out_path,
            coords=pos_np,
            times=data.time_stamps.numpy(),
            gt_temp=gt_temp,
            pred_temp=pred_temp,
            abs_err=abs_err,
            sim_id=data.sim_id,
            geometry=GEOMETRY_NAME,
        )

        nmse = ((pred_temp - gt_temp) ** 2).mean() / gt_temp.var().clip(min=1e-8)
        mae = abs_err.mean()
        rel_l2 = np.linalg.norm(pred_temp - gt_temp) / np.linalg.norm(gt_temp).clip(min=1e-8)
        max_err = abs_err.max()

        summary_rows.append({
            "sim_id": data.sim_id,
            "n_nodes": n_nodes,
            "n_timesteps": n_steps,
            "mean_abs_err_K": float(mae),
            "max_abs_err_K": float(max_err),
            "C1_normalized_mse": float(nmse),
            "C3_relative_L2": float(rel_l2),
            "infer_time_s": infer_time,
            "npz_path": out_path,
        })
        print(f"OK  {data.sim_id}  mean_err={mae:.3f}K  max_err={max_err:.3f}K")

    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, f"inference_summary_unet_{GEOMETRY_NAME}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved batch summary: {summary_csv}")


if __name__ == "__main__":
    main()