"""
train_mlp.py -- Stepped_Block

WHAT THIS DOES
---------------
Trains the pointwise MLP baseline on Stepped_Block, then evaluates it
against your six criteria (C1-C6).

Same architecture, data pipeline, checkpointing, and crash-safety
logic as the Brake/Flange versions -- only geometry name and paths
change. See mlp_model.py's docstring for full explanation of the
on-the-fly batch dataset, and the module docstring in the Flange
version of this script for the full explanation of the Kaggle
background-commit + checkpoint-resume workflow (same applies here).

SAVE VERIFICATION + CRASH SAFETY
-----------------------------------
- Every save is ATOMIC (write to .tmp, then os.replace).
- Every save prints an explicit [CHECK] line with file size and
  existence -- trust these log lines over the Output tab's summary
  widget, which has been observed to show 0B even when saves succeed.
- The epoch loop is wrapped in try/except: on ANY crash, an
  emergency_checkpoint.pt is saved before the exception propagates.

RESUMING ACROSS COMMITS: each new "Save Version -> Run All" starts
from a clean container and will NOT automatically see a previous
commit's /kaggle/working/ files. To truly resume: download the
previous run's checkpoint_stepped_block.pt, re-upload it as a Kaggle
input Dataset, add it as an Input to this notebook, and set
PREVIOUS_CHECKPOINT_INPUT below to its path before committing again.

Requires: pip install torch pandas numpy
"""

import os
import time
import datetime
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader



GEOMETRY_FAMILY = "Stepped_Block"
SPLITS_DIR = r"/kaggle/input/datasets/prismvet/stepped-block-splits"
RESULTS_DIR = r"/kaggle/working/stepped_block_results/mlp"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"[CHECK] RESULTS_DIR exists: {os.path.isdir(RESULTS_DIR)} -> {RESULTS_DIR}")

MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "mlp_stepped_block.pt")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint_stepped_block.pt")
EMERGENCY_CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "emergency_checkpoint.pt")
NORM_STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats_stepped_block.pt")
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics_stepped_block.csv")

# If you've re-uploaded a checkpoint from a previous commit as a Kaggle
# input dataset, point this at it. Leave as-is (nonexistent) for a
# first run -- the script will just start from epoch 1.
PREVIOUS_CHECKPOINT_INPUT = "/kaggle/input/datasets/prismvet/results-stepped-block/checkpoint_stepped_block.pt"
PREVIOUS_NORM_STATS_INPUT = "/kaggle/input/datasets/prismvet/results-stepped-block/norm_stats_stepped_block.pt"

BATCH_SIZE = 4096
N_EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_WORKERS = 0     # keep 0 -- data is fully in RAM; extra workers only
                    # duplicate memory via fork, no speed benefit here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(name):
    return torch.load(os.path.join(SPLITS_DIR, f"{name}.pt"), weights_only=False)


def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def count_total_rows(data_list):
    total = 0
    for data in data_list:
        total += data.x.shape[0] * data.y.shape[1]
    return total


def find_checkpoint_to_resume_from():
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    if os.path.exists(PREVIOUS_CHECKPOINT_INPUT):
        print(f"[CHECK] Found previous-commit checkpoint input: {PREVIOUS_CHECKPOINT_INPUT}")
        return PREVIOUS_CHECKPOINT_INPUT
    return None


def atomic_save(obj, final_path):
    tmp_path = final_path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)


def train_mlp(train_list, val_list):
    print(f"Using device: {DEVICE}")

    if os.path.exists(NORM_STATS_PATH):
        print("Found existing normalization stats -- reusing.")
        stats = torch.load(NORM_STATS_PATH, weights_only=False)
    elif os.path.exists(PREVIOUS_NORM_STATS_INPUT):
        print(f"Loading normalization stats from previous commit's input: {PREVIOUS_NORM_STATS_INPUT}")
        stats = torch.load(PREVIOUS_NORM_STATS_INPUT, weights_only=False)
        atomic_save(stats, NORM_STATS_PATH)
        print(f"[CHECK] Norm stats copied to working dir: exists={os.path.exists(NORM_STATS_PATH)} -> {NORM_STATS_PATH}")
    else:
        print("Computing normalization stats (streaming, one variant at a time)...")
        stats = compute_norm_stats_streaming(train_list)
        atomic_save(stats, NORM_STATS_PATH)
        print(f"[CHECK] Norm stats saved: exists={os.path.exists(NORM_STATS_PATH)} -> {NORM_STATS_PATH}")

    train_rows = count_total_rows(train_list)
    val_rows = count_total_rows(val_list)
    print(f"[train] total rows = {train_rows:,}   [val] total rows = {val_rows:,}")

    train_ds = OnTheFlyBatchDataset(train_list, stats, BATCH_SIZE)
    val_ds = OnTheFlyBatchDataset(val_list, stats, BATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=NUM_WORKERS)

    model = TemperatureMLP().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}, Total trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    start_epoch = 1
    best_val = float("inf")

    resume_path = find_checkpoint_to_resume_from()
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"Resuming from checkpoint ({resume_path}): epoch {start_epoch}, best_val={best_val:.6f}")
    else:
        print("No checkpoint found -- starting from epoch 1.")

    if start_epoch > N_EPOCHS:
        print("Training already complete per checkpoint. Skipping to evaluation.")
        return stats

    epoch_durations = []
    training_start = time.perf_counter()
    epoch = start_epoch

    try:
        for epoch in range(start_epoch, N_EPOCHS + 1):
            epoch_start = time.perf_counter()

            model.train()
            running, n_seen = 0.0, 0
            batch_count = 0
            batch_print_interval = 50
            loop_start = time.perf_counter()

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.shape[0]
                n_seen += xb.shape[0]
                batch_count += 1

                if batch_count % batch_print_interval == 0:
                    elapsed = time.perf_counter() - loop_start
                    rows_per_sec = n_seen / elapsed
                    pct_done = 100 * n_seen / train_rows
                    print(f"  [epoch {epoch}] batch {batch_count}  "
                          f"rows_seen={n_seen:,}/{train_rows:,} ({pct_done:.1f}%)  "
                          f"rows/sec={rows_per_sec:,.0f}")

            running /= max(n_seen, 1)

            model.eval()
            val_running, val_seen = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_running += loss_fn(model(xb), yb).item() * xb.shape[0]
                    val_seen += xb.shape[0]
            val_loss = val_running / max(val_seen, 1)

            epoch_duration = time.perf_counter() - epoch_start
            epoch_durations.append(epoch_duration)
            avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
            epochs_remaining = N_EPOCHS - epoch
            eta_seconds = avg_epoch_time * epochs_remaining

            print(f"Epoch {epoch:3d}/{N_EPOCHS}  train={running:.6f}  val={val_loss:.6f}  "
                  f"| epoch_time={format_duration(epoch_duration)}  "
                  f"avg_epoch_time={format_duration(avg_epoch_time)}  "
                  f"ETA={format_duration(eta_seconds)}")

            ckpt_data = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
            }
            atomic_save(ckpt_data, CHECKPOINT_PATH)
            ckpt_size_mb = os.path.getsize(CHECKPOINT_PATH) / 1e6
            print(f"[CHECK] Checkpoint saved: epoch={epoch}  size={ckpt_size_mb:.2f}MB  "
                  f"exists={os.path.exists(CHECKPOINT_PATH)}  -> {CHECKPOINT_PATH}")

            if val_loss < best_val:
                best_val = val_loss
                atomic_save(model.state_dict(), MODEL_SAVE_PATH)
                model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / 1e6
                print(f"[CHECK] New best model saved: val_loss={val_loss:.6f}  "
                      f"size={model_size_mb:.2f}MB  -> {MODEL_SAVE_PATH}")

    except Exception as e:
        print(f"\n[ERROR] Training crashed at epoch {epoch}: {e}")
        traceback.print_exc()
        try:
            atomic_save({
                "epoch": epoch - 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
            }, EMERGENCY_CHECKPOINT_PATH)
            print(f"[CHECK] Emergency checkpoint saved -> {EMERGENCY_CHECKPOINT_PATH}")
        except Exception as save_err:
            print(f"[ERROR] Even emergency save failed: {save_err}")
        raise

    total_time = time.perf_counter() - training_start
    print(f"\nTraining complete. Total time: {format_duration(total_time)}")
    print(f"Best val_loss={best_val:.6f} -> {MODEL_SAVE_PATH}")

    print("\n[CHECK] Final contents of RESULTS_DIR:")
    for f in os.listdir(RESULTS_DIR):
        full_path = os.path.join(RESULTS_DIR, f)
        size_mb = os.path.getsize(full_path) / 1e6
        print(f"  {f}  ({size_mb:.2f} MB)")

    return stats


def evaluate_mlp(test_list, stats):
    model = TemperatureMLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    rows = []
    for data in test_list:
        n_nodes, n_steps = data.x.shape[0], data.y.shape[1]
        static_rep = data.x.unsqueeze(1).expand(n_nodes, n_steps, 6)
        time_rep = data.time_stamps.view(1, n_steps, 1).expand(n_nodes, n_steps, 1)
        inputs = torch.cat([static_rep, time_rep], dim=2).reshape(n_nodes * n_steps, 7)
        inputs_n = ((inputs - stats["input_mean"]) / stats["input_std"]).to(DEVICE)

        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_n = model(inputs_n)
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        infer_time = time.perf_counter() - t0

        pred = denormalize_targets(pred_n.cpu(), stats).reshape(n_nodes, n_steps)
        true = data.y
        err = pred - true
        abs_err = err.abs()

        nmse = (err ** 2).mean().item() / true.var().clamp_min(1e-8).item()
        mae = abs_err.mean().item()
        rel_l2 = (torch.norm(err) / torch.norm(true).clamp_min(1e-8)).item()
        max_err = abs_err.max().item()
        p95_err = torch.quantile(abs_err, 0.95).item()

        true_curve = true.mean(dim=0)
        pred_curve = pred.mean(dim=0)
        time_avg_abs_err = (true_curve - pred_curve).abs().mean().item()
        tc, pc = true_curve - true_curve.mean(), pred_curve - pred_curve.mean()
        temporal_corr = (tc @ pc / (tc.norm() * pc.norm()).clamp_min(1e-8)).item()

        rows.append({
            "sim_id": data.sim_id, "n_nodes": n_nodes, "n_timesteps": n_steps,
            "C1_normalized_mse": nmse, "C2_mae_K": mae, "C3_relative_L2": rel_l2,
            "C4_max_abs_err_K": max_err, "C4_p95_abs_err_K": p95_err,
            "C5_time_avg_abs_err_K": time_avg_abs_err, "C5_temporal_correlation": temporal_corr,
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
    print(f"[CHECK] Metrics CSV saved: exists={os.path.exists(METRICS_CSV_PATH)} -> {METRICS_CSV_PATH}")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    train_list = load_split("train")
    val_list = load_split("val")
    test_list = load_split("test")
    stats = train_mlp(train_list, val_list)
    evaluate_mlp(test_list, stats)