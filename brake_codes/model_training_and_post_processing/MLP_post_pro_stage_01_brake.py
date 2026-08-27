"""
STAGE 1 (BATCH): Run inference for every processed variant .pt file of a
given geometry family, save one standardized .npz per variant, and build
a summary CSV of node/time-resolved errors so you can select which
variants to visualize in Stage 2 (e.g. best / median / worst).
"""

import os
import glob
import numpy as np
import pandas as pd
import torch

# ---- EDIT THESE ----
GEOMETRY_NAME    = "brake"                          # "brake" | "flange" | "stepped_block"
GRAPH_PT_DIR     = r"E:\NeurIPS_dataset\brake_models\brake_data_pt"   # where your builder saved the .pt files
GRAPH_PT_GLOB    = "Brake_*.pt"                      # match pattern for this geometry's variants
MODEL_PATH  = r"E:\NeurIPS_dataset\brake_models\brake_results\mlp\mlp_brake.pt"
NORM_STATS_PATH  = r"E:\NeurIPS_dataset\brake_models\brake_results\mlp\norm_stats_brake.pt"
OUT_DIR          = r"E:\NeurIPS_dataset\brake_models\field_data_brake"
# ---------------------

os.makedirs(OUT_DIR, exist_ok=True)


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


def build_input_tensor(pos, sdf, pour_temp, mould_temp, time_stamps):
    n_nodes = pos.shape[0]
    n_t = time_stamps.shape[0]
    pos_rep = pos.unsqueeze(1).expand(n_nodes, n_t, 3)
    sdf_rep = sdf.view(n_nodes, 1, 1).expand(n_nodes, n_t, 1)
    pour_rep = torch.full((n_nodes, n_t, 1), float(pour_temp))
    mould_rep = torch.full((n_nodes, n_t, 1), float(mould_temp))
    time_rep = time_stamps.view(1, n_t, 1).expand(n_nodes, n_t, 1)
    feats = torch.cat([pos_rep, sdf_rep, pour_rep, mould_rep, time_rep], dim=-1)
    return feats.reshape(n_nodes * n_t, 7), n_nodes, n_t


def run_one_variant(pt_path, model, in_mean, in_std, out_mean, out_std):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    pos = data["pos"]
    sdf = data["sdf"]
    temp_history = data["temp_history"]
    time_stamps = data["time_stamps"]
    pour_temp = data["pour_temp"]
    mould_temp = data["mould_temp"]

    raw_feats, n_nodes, n_t = build_input_tensor(pos, sdf, pour_temp, mould_temp, time_stamps)
    x_norm = (raw_feats - in_mean) / in_std

    with torch.no_grad():
        pred_norm = model(x_norm)
    pred_temp = (pred_norm * out_std + out_mean).reshape(n_nodes, n_t).numpy()

    gt_temp = temp_history.numpy()
    abs_err = np.abs(pred_temp - gt_temp)

    variant_name = os.path.splitext(os.path.basename(pt_path))[0]
    out_path = os.path.join(OUT_DIR, f"field_data_{variant_name}.npz")

    np.savez(
        out_path,
        coords=pos.numpy(),
        times=time_stamps.numpy(),
        gt_temp=gt_temp,
        pred_temp=pred_temp,
        abs_err=abs_err,
        sim_id=variant_name,
        geometry=GEOMETRY_NAME,
    )

    return {
        "sim_id": variant_name,
        "n_nodes": n_nodes,
        "n_timesteps": n_t,
        "pour_temp_K": float(pour_temp),
        "mould_temp_K": float(mould_temp),
        "mean_abs_err_K": float(abs_err.mean()),
        "max_abs_err_K": float(abs_err.max()),
        "npz_path": out_path,
    }


def main():
    model = MLP(in_dim=7, out_dim=1)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    norm = torch.load(NORM_STATS_PATH, map_location="cpu", weights_only=False)
    in_mean, in_std = norm["input_mean"], norm["input_std"]
    out_mean, out_std = norm["target_mean"], norm["target_std"]

    pt_files = sorted(glob.glob(os.path.join(GRAPH_PT_DIR, GRAPH_PT_GLOB)))
    print(f"Found {len(pt_files)} variant .pt files for {GEOMETRY_NAME}")

    summary_rows = []
    for pt_path in pt_files:
        try:
            row = run_one_variant(pt_path, model, in_mean, in_std, out_mean, out_std)
            summary_rows.append(row)
            print(f"OK  {row['sim_id']}  mean_err={row['mean_abs_err_K']:.3f}K  "
                  f"max_err={row['max_abs_err_K']:.3f}K")
        except Exception as e:
            print(f"FAIL {pt_path}: {e}")

    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, f"inference_summary_{GEOMETRY_NAME}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved batch summary: {summary_csv}")

    # ---- Suggest best / median / worst variants for Stage 2 ----
    if len(summary) > 0:
        s_sorted = summary.sort_values("mean_abs_err_K").reset_index(drop=True)
        best = s_sorted.iloc[0]
        worst = s_sorted.iloc[-1]
        median = s_sorted.iloc[len(s_sorted) // 2]

        print("\n=== Suggested variants to visualize in Stage 2 ===")
        print(f"  BEST   : {best['sim_id']}   (mean err = {best['mean_abs_err_K']:.3f} K)")
        print(f"  MEDIAN : {median['sim_id']}   (mean err = {median['mean_abs_err_K']:.3f} K)")
        print(f"  WORST  : {worst['sim_id']}   (mean err = {worst['mean_abs_err_K']:.3f} K)")


if __name__ == "__main__":
    main()