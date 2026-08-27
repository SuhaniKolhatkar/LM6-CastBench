"""
train_mlp.py

WHAT THIS DOES
---------------
Trains the pointwise MLP baseline on Brake, then evaluates it against
your six criteria (C1-C6).

SPEED FIX: uses ChunkedMemmapDataset (see mlp_model.py) which reads
large sequential blocks off disk instead of one random row at a time
-- the earlier version was effectively unusable on an external HDD.

RESUME SUPPORT: after every epoch, model weights + optimizer state +
current epoch number are checkpointed to disk. If training is
interrupted (crash, power loss, Ctrl+C), rerunning this script will
automatically pick up from the last completed epoch instead of
starting over from epoch 1.

ETA: after each epoch, prints elapsed time for that epoch, average
epoch time so far, and an estimated time remaining until N_EPOCHS.

Run AFTER build_brake_splits.py. This is Model 1 of 3.

Requires: pip install torch pandas numpy
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mlp_model import (
    TemperatureMLP,
    build_pointwise_dataset_to_disk,
    compute_norm_stats_streaming,
    ChunkedMemmapDataset,
    denormalize_targets,
)

GEOMETRY_FAMILY = "Brake"
SPLITS_DIR = r"E:\NeurIPS_dataset\brake_models\brake_splits"
RESULTS_DIR = r"E:\NeurIPS_dataset\brake_models\brake_results\mlp"
MEMMAP_DIR = r"E:\NeurIPS_dataset\brake_models\brake_results\mlp\memmap_cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MEMMAP_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "mlp_brake.pt")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint_brake.pt")
NORM_STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats_brake.pt")
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics_brake.csv")

BATCH_SIZE = 4096
N_EPOCHS = 50
LEARNING_RATE = 1e-3
CHUNK_SIZE = 200_000     # rows read per sequential disk block
NUM_WORKERS = 0          # keep 0 on Windows -- avoids the memmap pickling error
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(name):
    return torch.load(os.path.join(SPLITS_DIR, f"{name}.pt"), weights_only=False)


def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def train_mlp(train_list, val_list):
    print(f"Using device: {DEVICE}")

    if os.path.exists(NORM_STATS_PATH):
        print("Found existing normalization stats -- reusing.")
        stats = torch.load(NORM_STATS_PATH, weights_only=False)
    else:
        print("Computing normalization stats (streaming, one variant at a time)...")
        stats = compute_norm_stats_streaming(train_list)
        torch.save(stats, NORM_STATS_PATH)

    print("Preparing flattened train/val rows on disk (memmap)...")
    train_in_path, train_tgt_path, train_rows = build_pointwise_dataset_to_disk(
        train_list, MEMMAP_DIR, "train"
    )
    val_in_path, val_tgt_path, val_rows = build_pointwise_dataset_to_disk(
        val_list, MEMMAP_DIR, "val"
    )

    train_ds = ChunkedMemmapDataset(train_in_path, train_tgt_path, train_rows, stats, CHUNK_SIZE)
    val_ds = ChunkedMemmapDataset(val_in_path, val_tgt_path, val_rows, stats, CHUNK_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = TemperatureMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    start_epoch = 1
    best_val = float("inf")

    # --- Resume from checkpoint if one exists ---
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"Resuming from checkpoint: epoch {start_epoch}, best_val={best_val:.6f}")

    if start_epoch > N_EPOCHS:
        print("Training already complete per checkpoint. Skipping to evaluation.")
        return stats

    epoch_durations = []
    training_start = time.perf_counter()

    for epoch in range(start_epoch, N_EPOCHS + 1):
        epoch_start = time.perf_counter()

        model.train()
        running, n_seen = 0.0, 0
        batch_count = 0
        batch_print_interval = 200   # print progress every 200 batches
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

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Checkpoint EVERY epoch (not just best) so a crash never loses more
        # than the current epoch's progress
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
        }, CHECKPOINT_PATH)

    total_time = time.perf_counter() - training_start
    print(f"\nTraining complete. Total time: {format_duration(total_time)}")
    print(f"Best val_loss={best_val:.6f} -> {MODEL_SAVE_PATH}")
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
    print(f"\nSaved metrics -> {METRICS_CSV_PATH}")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    train_list = load_split("train")
    val_list = load_split("val")
    test_list = load_split("test")
    stats = train_mlp(train_list, val_list)
    evaluate_mlp(test_list, stats)