"""
run_evaluation_from_checkpoint.py

WHAT THIS DOES
---------------
Standalone evaluator: loads a TRAINED checkpoint (model_state dict
inside a checkpoint_*.pt, or a plain state_dict from mlp_*.pt) plus
its matching norm_stats_*.pt, runs inference against a geometry's
test.pt split, and computes the same C1-C6 metrics + writes the same
metrics_<geometry>.csv format that evaluate_mlp() produces during
training -- so you can regenerate this file for Flange/Stepped_Block
now, without re-running training.

Uses BATCHED inference (see earlier fix) to avoid GPU/CPU OOM on
large meshes.

Requires: pip install torch pandas numpy
"""

import os
import time
import numpy as np
import pandas as pd
import torch

from mlp_model import TemperatureMLP, denormalize_targets

# ---------------------------------------------------------------------
# CONFIG -- fill these in per geometry, run once per geometry
# ---------------------------------------------------------------------
GEOMETRY_FAMILY = "Flange"          # change to "Stepped_Block" for the other run
SPLITS_DIR = r"E:\NeurIPS_dataset\flange_models\flange_splits"        # must contain test.pt
CHECKPOINT_PATH = r"E:\NeurIPS_dataset\flange_models\flange_results\mlp\checkpoint_flange_post.pt"   # or mlp_flange.pt
NORM_STATS_PATH = r"E:\NeurIPS_dataset\flange_models\flange_results\mlp\norm_stats_flange_post.pt"
OUTPUT_DIR = r"E:\NeurIPS_dataset\flange_models\flange_results\mlp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, f"metrics_{GEOMETRY_FAMILY.lower()}.csv")

BATCH_SIZE = 8192   # inference batch size, safe for limited VRAM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, device):
    """
    Handles BOTH formats you might have:
      - a full checkpoint dict with a "model_state" key
      - a plain state_dict (e.g. mlp_flange.pt, saved via
        torch.save(model.state_dict(), ...))
    """
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TemperatureMLP().to(device)

    if isinstance(obj, dict) and "model_state" in obj:
        model.load_state_dict(obj["model_state"])
        print(f"Loaded from checkpoint dict. epoch={obj.get('epoch')}  "
              f"best_val={obj.get('best_val')}")
    else:
        model.load_state_dict(obj)
        print("Loaded from plain state_dict file.")

    model.eval()
    return model


def run_inference_batched(model, stats, data, batch_size, device):
    n_nodes, n_steps = data.x.shape[0], data.y.shape[1]
    static_rep = data.x.unsqueeze(1).expand(n_nodes, n_steps, 6)
    time_rep = data.time_stamps.view(1, n_steps, 1).expand(n_nodes, n_steps, 1)
    inputs = torch.cat([static_rep, time_rep], dim=2).reshape(n_nodes * n_steps, 7)
    inputs_n = (inputs - stats["input_mean"]) / stats["input_std"]   # stays on CPU

    total_rows = inputs_n.shape[0]
    pred_chunks = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = inputs_n[start:end].to(device)
            pred_batch_n = model(batch)
            pred_chunks.append(pred_batch_n.cpu())
            del batch, pred_batch_n
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - t0

    pred_n = torch.cat(pred_chunks, dim=0)
    pred = denormalize_targets(pred_n, stats).reshape(n_nodes, n_steps)
    true = data.y
    return pred, true, infer_time


def evaluate(model, stats, test_list):
    rows = []
    for data in test_list:
        pred, true, infer_time = run_inference_batched(model, stats, data, BATCH_SIZE, DEVICE)

        err = pred - true
        abs_err = err.abs()

        nmse = (err ** 2).mean().item() / true.var().clamp_min(1e-8).item()
        mae = abs_err.mean().item()
        rel_l2 = (torch.norm(err) / torch.norm(true).clamp_min(1e-8)).item()
        max_err = abs_err.max().item()
        p95_err = float(np.percentile(abs_err.numpy(), 95))

        true_curve = true.mean(dim=0)
        pred_curve = pred.mean(dim=0)
        time_avg_abs_err = (true_curve - pred_curve).abs().mean().item()
        tc, pc = true_curve - true_curve.mean(), pred_curve - pred_curve.mean()
        temporal_corr = (tc @ pc / (tc.norm() * pc.norm()).clamp_min(1e-8)).item()

        rows.append({
            "sim_id": data.sim_id,
            "n_nodes": data.x.shape[0],
            "n_timesteps": data.y.shape[1],
            "C1_normalized_mse": nmse,
            "C2_mae_K": mae,
            "C3_relative_L2": rel_l2,
            "C4_max_abs_err_K": max_err,
            "C4_p95_abs_err_K": p95_err,
            "C5_time_avg_abs_err_K": time_avg_abs_err,
            "C5_temporal_correlation": temporal_corr,
            "C6_inference_time_s": infer_time,
        })
        print(f"{data.sim_id}: NMSE={nmse:.5f} MAE={mae:.3f}K relL2={rel_l2:.4f} "
              f"max={max_err:.2f}K corr={temporal_corr:.4f} t={infer_time*1000:.2f}ms")

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c.startswith("C")]
    avg = df[numeric_cols].mean()
    avg["sim_id"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"\nSaved metrics -> {METRICS_CSV_PATH}")
    print(df.tail(1).to_string(index=False))
    return df


if __name__ == "__main__":
    stats = torch.load(NORM_STATS_PATH, weights_only=False)
    model = load_model(CHECKPOINT_PATH, DEVICE)
    test_list = torch.load(os.path.join(SPLITS_DIR, "test.pt"), weights_only=False)
    evaluate(model, stats, test_list)