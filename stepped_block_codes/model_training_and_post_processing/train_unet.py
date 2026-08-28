"""
train_unet.py -- stepped_block

WHAT THIS DOES
---------------
Trains the 3D U-Net on Brake: regrids each variant's mesh onto a
GRID_RES^3 voxel grid (see grid_utils.py), trains per-timestep (same
"time as an explicit input channel" strategy as the MLP/GNO models),
and evaluates by UN-PROJECTING grid predictions back onto the
original mesh nodes -- so C1-C6 metrics are computed on the exact
same basis as your MLP/GNO results, for a fair benchmark comparison.

TO RUN FOR FLANGE / STEPPED_BLOCK: copy this file, change only the
CONFIG block below (GEOMETRY_FAMILY, SPLITS_DIR, RESULTS_DIR, and the
PREVIOUS_CHECKPOINT_INPUT path if resuming). Nothing else needs to
change -- grid_utils.py and unet_model.py are fully geometry-agnostic.

WHY GRID REGRIDDING SOLVES A PROBLEM MLP HAD
------------------------------------------------
Unlike the MLP's flattened dataset (which grew to 60GB+ for Flange
and blew Kaggle's disk/RAM limits), U-Net's per-sample size is FIXED
regardless of mesh size: every variant becomes exactly
[5, GRID_RES, GRID_RES, GRID_RES] no matter how many real mesh nodes
it has. This means Flange's larger mesh causes NO extra memory
pressure here -- a major practical advantage of the grid approach.

KAGGLE BACKGROUND-COMMIT + RESUME WORKFLOW: identical strategy to the
MLP scripts -- atomic checkpoint saves every epoch, [CHECK] log lines
to verify saves succeeded, try/except with an emergency checkpoint on
crash, and PREVIOUS_CHECKPOINT_INPUT for resuming across separate
"Save Version -> Run All" commits (each commit starts fresh and does
NOT automatically see a previous commit's /kaggle/working/ files --
you must re-upload the checkpoint as an input dataset between commits).

Requires: pip install torch scipy numpy pandas
"""

import os
import time
import datetime
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

#from grid_utils import build_grid_mapping, build_input_grid, build_target_grid, unproject_grid_to_nodes
#from unet_model import TemperatureUNet3D

# ---------------------------------------------------------------------
# CONFIG -- only this block changes between Brake / Flange / Stepped_Block
# ---------------------------------------------------------------------
GEOMETRY_FAMILY = "stepped_block"
SPLITS_DIR = r"/kaggle/input/datasets/probuildvet/stepped-block-split-data"
RESULTS_DIR = r"/kaggle/working/stepped_block_results/unet"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"[CHECK] RESULTS_DIR exists: {os.path.isdir(RESULTS_DIR)} -> {RESULTS_DIR}")

MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "unet_stepped_block.pt")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint_unet_stepped_block.pt")
EMERGENCY_CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "emergency_checkpoint.pt")
NORM_STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats_unet_stepped_block.pt")
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics_unet_stepped_block.csv")

# For resuming across separate Kaggle commits -- update after re-uploading
PREVIOUS_CHECKPOINT_INPUT = "/kaggle/input/datasets/probuildvet/stepped-block-results/checkpoint_unet_stepped_block.pt"

GRID_RES = 32          # voxel grid resolution per axis -- lower (e.g. 24) if VRAM-limited
BATCH_SIZE = 8          # number of (variant, timestep) grids per batch
N_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(name):
    return torch.load(os.path.join(SPLITS_DIR, f"{name}.pt"), weights_only=False)


def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def atomic_save(obj, final_path):
    tmp_path = final_path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)


def find_checkpoint_to_resume_from():
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    if os.path.exists(PREVIOUS_CHECKPOINT_INPUT):
        print(f"[CHECK] Found previous-commit checkpoint input: {PREVIOUS_CHECKPOINT_INPUT}")
        return PREVIOUS_CHECKPOINT_INPUT
    return None


# ---------------------------------------------------------------------
# Precompute grid mappings ONCE per variant (mesh geometry doesn't
# change across epochs -- only temperature values do)
# ---------------------------------------------------------------------
def precompute_mappings(data_list, grid_res):
    mappings = []
    for i, data in enumerate(data_list):
        pos_np = data.pos.numpy()
        mapping = build_grid_mapping(pos_np, grid_res)
        mappings.append(mapping)
        if (i + 1) % 20 == 0 or (i + 1) == len(data_list):
            print(f"  Precomputed grid mapping {i+1}/{len(data_list)}")
    return mappings


# ---------------------------------------------------------------------
# Normalization stats -- streamed over all (variant, timestep) grids
# ---------------------------------------------------------------------
def compute_norm_stats_streaming(data_list, mappings, grid_res):
    input_sum = torch.zeros(5, dtype=torch.float64)
    input_sumsq = torch.zeros(5, dtype=torch.float64)
    target_sum = torch.zeros(1, dtype=torch.float64)
    target_sumsq = torch.zeros(1, dtype=torch.float64)
    total_voxels = 0

    for data, mapping in zip(data_list, mappings):
        sdf_np = data.sdf.numpy()
        temp_np = data.y.numpy()
        n_steps = temp_np.shape[1]

        for t in range(n_steps):
            time_val = float(data.time_stamps[t])
            x_grid = build_input_grid(mapping, sdf_np, float(data.pour_temp),
                                       float(data.mould_temp), time_val, grid_res)
            y_grid = build_target_grid(mapping, temp_np[:, t], grid_res)

            x_flat = x_grid.reshape(5, -1).double()
            y_flat = y_grid.reshape(1, -1).double()

            input_sum += x_flat.sum(dim=1)
            input_sumsq += (x_flat ** 2).sum(dim=1)
            target_sum += y_flat.sum(dim=1)
            target_sumsq += (y_flat ** 2).sum(dim=1)
            total_voxels += x_flat.shape[1]

    input_mean = (input_sum / total_voxels).float()
    input_var = (input_sumsq / total_voxels).float() - input_mean ** 2
    input_std = input_var.clamp_min(1e-8).sqrt()

    target_mean = (target_sum / total_voxels).float()
    target_var = (target_sumsq / total_voxels).float() - target_mean ** 2
    target_std = target_var.clamp_min(1e-8).sqrt()

    return {
        "input_mean": input_mean.view(1, 5, 1, 1, 1),
        "input_std": input_std.view(1, 5, 1, 1, 1),
        "target_mean": target_mean.view(1, 1, 1, 1, 1),
        "target_std": target_std.view(1, 1, 1, 1, 1),
    }


# ---------------------------------------------------------------------
# On-the-fly grid batch dataset -- fixed sample size regardless of
# mesh size (see module docstring: this is why U-Net sidesteps the
# disk/RAM blowup MLP had on Flange)
# ---------------------------------------------------------------------
class OnTheFlyGridDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_list, mappings, stats, grid_res, batch_size):
        self.data_list = data_list
        self.mappings = mappings
        self.stats = stats
        self.grid_res = grid_res
        self.batch_size = batch_size

    def __iter__(self):
        # Build a flat list of (variant_idx, timestep_idx) samples, shuffle
        samples = []
        for vi, data in enumerate(self.data_list):
            for t in range(data.y.shape[1]):
                samples.append((vi, t))
        np.random.shuffle(samples)

        batch_x, batch_y = [], []
        for vi, t in samples:
            data = self.data_list[vi]
            mapping = self.mappings[vi]
            sdf_np = data.sdf.numpy()
            temp_np = data.y.numpy()
            time_val = float(data.time_stamps[t])

            x_grid = build_input_grid(mapping, sdf_np, float(data.pour_temp),
                                       float(data.mould_temp), time_val, self.grid_res)
            y_grid = build_target_grid(mapping, temp_np[:, t], self.grid_res)

            x_n = (x_grid - self.stats["input_mean"].squeeze(0)) / self.stats["input_std"].squeeze(0)
            y_n = (y_grid - self.stats["target_mean"].squeeze(0)) / self.stats["target_std"].squeeze(0)

            batch_x.append(x_n)
            batch_y.append(y_n)

            if len(batch_x) == self.batch_size:
                yield torch.stack(batch_x), torch.stack(batch_y)
                batch_x, batch_y = [], []

        if batch_x:
            yield torch.stack(batch_x), torch.stack(batch_y)


def train_unet(train_list, val_list):
    print(f"Using device: {DEVICE}")

    print("Precomputing grid mappings for train set...")
    train_mappings = precompute_mappings(train_list, GRID_RES)
    print("Precomputing grid mappings for val set...")
    val_mappings = precompute_mappings(val_list, GRID_RES)

    if os.path.exists(NORM_STATS_PATH):
        print("Found existing normalization stats -- reusing.")
        stats = torch.load(NORM_STATS_PATH, weights_only=False)
    else:
        print("Computing normalization stats (streaming over all variant/timestep grids)...")
        stats = compute_norm_stats_streaming(train_list, train_mappings, GRID_RES)
        atomic_save(stats, NORM_STATS_PATH)
        print(f"[CHECK] Norm stats saved: exists={os.path.exists(NORM_STATS_PATH)} -> {NORM_STATS_PATH}")

    train_ds = OnTheFlyGridDataset(train_list, train_mappings, stats, GRID_RES, BATCH_SIZE)
    val_ds = OnTheFlyGridDataset(val_list, val_mappings, stats, GRID_RES, BATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=0)

    model = TemperatureUNet3D().to(DEVICE)
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
        return stats, train_mappings, val_mappings

    epoch_durations = []
    training_start = time.perf_counter()
    epoch = start_epoch

    try:
        for epoch in range(start_epoch, N_EPOCHS + 1):
            epoch_start = time.perf_counter()

            model.train()
            running, n_seen = 0.0, 0
            batch_count = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.shape[0]
                n_seen += xb.shape[0]
                batch_count += 1
                if batch_count % 50 == 0:
                    print(f"  [epoch {epoch}] batch {batch_count}  samples_seen={n_seen}")
            running /= max(n_seen, 1)

            model.eval()
            val_running, val_seen = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb)
                    val_running += loss_fn(pred, yb).item() * xb.shape[0]
                    val_seen += xb.shape[0]
            val_loss = val_running / max(val_seen, 1)

            epoch_duration = time.perf_counter() - epoch_start
            epoch_durations.append(epoch_duration)
            avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
            eta_seconds = avg_epoch_time * (N_EPOCHS - epoch)

            print(f"Epoch {epoch:3d}/{N_EPOCHS}  train={running:.6f}  val={val_loss:.6f}  "
                  f"| epoch_time={format_duration(epoch_duration)}  "
                  f"avg_epoch_time={format_duration(avg_epoch_time)}  "
                  f"ETA={format_duration(eta_seconds)}")

            ckpt_data = {
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "best_val": best_val,
            }
            atomic_save(ckpt_data, CHECKPOINT_PATH)
            print(f"[CHECK] Checkpoint saved: epoch={epoch}  "
                  f"size={os.path.getsize(CHECKPOINT_PATH)/1e6:.2f}MB -> {CHECKPOINT_PATH}")

            if val_loss < best_val:
                best_val = val_loss
                atomic_save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"[CHECK] New best model saved: val_loss={val_loss:.6f} -> {MODEL_SAVE_PATH}")

    except Exception as e:
        print(f"\n[ERROR] Training crashed at epoch {epoch}: {e}")
        traceback.print_exc()
        try:
            atomic_save({
                "epoch": epoch - 1, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "best_val": best_val,
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
        print(f"  {f}  ({os.path.getsize(os.path.join(RESULTS_DIR, f))/1e6:.2f} MB)")

    return stats, train_mappings, val_mappings


def evaluate_unet(test_list, stats):
    print("Precomputing grid mappings for test set...")
    test_mappings = precompute_mappings(test_list, GRID_RES)

    model = TemperatureUNet3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    rows = []
    for data, mapping in zip(test_list, test_mappings):
        n_nodes = data.pos.shape[0]
        n_steps = data.y.shape[1]
        sdf_np = data.sdf.numpy()
        true_full = data.y  # [N, T]
        pred_full = torch.zeros(n_nodes, n_steps)

        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(n_steps):
                time_val = float(data.time_stamps[t])
                x_grid = build_input_grid(mapping, sdf_np, float(data.pour_temp),
                                           float(data.mould_temp), time_val, GRID_RES)
                x_n = (x_grid - stats["input_mean"].squeeze(0)) / stats["input_std"].squeeze(0)
                x_n = x_n.unsqueeze(0).to(DEVICE)  # add batch dim

                pred_grid_n = model(x_n).squeeze(0).squeeze(0).cpu()  # [D,H,W]
                pred_grid = pred_grid_n * stats["target_std"].squeeze() + stats["target_mean"].squeeze()

                pred_nodes = unproject_grid_to_nodes(pred_grid.numpy(), mapping)
                pred_full[:, t] = torch.from_numpy(pred_nodes)
        infer_time = time.perf_counter() - t0

        err = pred_full - true_full
        abs_err = err.abs()

        nmse = (err ** 2).mean().item() / true_full.var().clamp_min(1e-8).item()
        mae = abs_err.mean().item()
        rel_l2 = (torch.norm(err) / torch.norm(true_full).clamp_min(1e-8)).item()
        max_err = abs_err.max().item()
        p95_err = float(np.percentile(abs_err.numpy(), 95))

        true_curve = true_full.mean(dim=0)
        pred_curve = pred_full.mean(dim=0)
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
    print(f"[CHECK] Metrics CSV saved -> {METRICS_CSV_PATH}")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    train_list = load_split("train")
    val_list = load_split("val")
    test_list = load_split("test")
    stats, train_mappings, val_mappings = train_unet(train_list, val_list)
    evaluate_unet(test_list, stats)