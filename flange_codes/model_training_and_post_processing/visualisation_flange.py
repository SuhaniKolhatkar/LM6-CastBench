"""
visualize_mlp_flange.py

WHAT THIS DOES
---------------
Loads the FULLY TRAINED (epoch 50/50) Flange MLP checkpoint and its
matching normalization stats, runs inference on test-set variants, and
produces the same four visualizations as the Brake script:

  1. 3D spatial field plots (PyVista): actual / predicted / abs error,
     rendered on real node positions, at a chosen timestep.
  2. Time-evolution curve: mean temperature vs. time, actual vs.
     predicted (visualizes C5).
  3. Error histogram at the chosen timestep (visualizes C4).
  4. Predicted-vs-true scatter plot (regression sanity check).

IMPORTANT: this script uses checkpoint_flange_post.pt and
norm_stats_flange_post.pt -- the POST-RESUME files, which reached the
full 50/50 epochs. The pre-resume checkpoint_flange.pt (stopped at
epoch 32) is superseded and not used here.

Run this wherever your Flange test.pt split file lives (e.g. your
Kaggle notebook, pointed at flange_splits input) -- it needs real
mesh/geometry data to visualize against, not just the checkpoint.

Requires: pip install pyvista matplotlib numpy torch
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyvista as pv

from mlp_model import TemperatureMLP, denormalize_targets

# ---------------------------------------------------------------------
# CONFIG -- adjust these paths to wherever you're running this
# ---------------------------------------------------------------------
GEOMETRY_FAMILY = "Flange"

# Point this at wherever your Flange test.pt split lives
SPLITS_DIR = r"E:\NeurIPS_dataset\flange_models\flange_splits"

# Point these at your downloaded/re-uploaded checkpoint files
CHECKPOINT_PATH = r"E:\NeurIPS_dataset\flange_models\flange_results\mlp\checkpoint_flange_post.pt"
NORM_STATS_PATH = r"E:\NeurIPS_dataset\flange_models\flange_results\mlp\norm_stats_flange_post.pt"

VIZ_DIR = "flange_visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANT_SIM_IDS = None      # e.g. ["Flange_p953_m303_v01"] to pick specific ones
N_VARIANTS_TO_PLOT = 2
TIMESTEP_INDEX_FOR_SPATIAL_PLOTS = -1   # -1 = last (fully cooled) timestep


# ---------------------------------------------------------------------
# Load model + stats from the CHECKPOINT (not a separately-saved
# "best model" file this time -- checkpoint_flange_post.pt has
# everything: model_state, optimizer_state, epoch).
# ---------------------------------------------------------------------
def load_model_and_stats():
    stats = torch.load(NORM_STATS_PATH, weights_only=False)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    print(f"Loaded checkpoint at epoch {ckpt['epoch']}, best_val={ckpt['best_val']:.6f}")

    model = TemperatureMLP().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stats


def run_inference(model, stats, data, batch_size=8192):
    """
    Runs inference in BATCHES instead of one giant tensor, to fit
    within limited GPU VRAM (e.g. a 6GB laptop GPU). Builds the full
    [N*T, 7] input on CPU (cheap, just RAM), then feeds it to the GPU
    in chunks, moving each batch's predictions back to CPU immediately
    so GPU memory never holds more than one batch at a time.
    """
    n_nodes, n_steps = data.x.shape[0], data.y.shape[1]
    static_rep = data.x.unsqueeze(1).expand(n_nodes, n_steps, 6)
    time_rep = data.time_stamps.view(1, n_steps, 1).expand(n_nodes, n_steps, 1)
    inputs = torch.cat([static_rep, time_rep], dim=2).reshape(n_nodes * n_steps, 7)
    inputs_n = (inputs - stats["input_mean"]) / stats["input_std"]   # stays on CPU

    total_rows = inputs_n.shape[0]
    pred_chunks = []

    with torch.no_grad():
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = inputs_n[start:end].to(DEVICE)
            pred_batch_n = model(batch)
            pred_chunks.append(pred_batch_n.cpu())   # move back to CPU immediately
            del batch, pred_batch_n
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    pred_n = torch.cat(pred_chunks, dim=0)
    pred = denormalize_targets(pred_n, stats).reshape(n_nodes, n_steps)
    true = data.y
    return pred, true


def plot_spatial_comparison(data, pred, true, t_idx, out_path):
    pos_np = data.pos.numpy()
    true_t = true[:, t_idx].numpy()
    pred_t = pred[:, t_idx].numpy()
    abs_err_t = np.abs(pred_t - true_t)
    physical_time = float(data.time_stamps[t_idx])

    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1800, 600))
    tmin, tmax = min(true_t.min(), pred_t.min()), max(true_t.max(), pred_t.max())

    cloud_true = pv.PolyData(pos_np)
    cloud_true["Temperature"] = true_t
    plotter.subplot(0, 0)
    plotter.add_mesh(cloud_true, scalars="Temperature", cmap="inferno",
                      clim=[tmin, tmax], point_size=6, render_points_as_spheres=True)
    plotter.add_text(f"Actual (t={physical_time:.2f}s)", font_size=10)

    cloud_pred = pv.PolyData(pos_np)
    cloud_pred["Temperature"] = pred_t
    plotter.subplot(0, 1)
    plotter.add_mesh(cloud_pred, scalars="Temperature", cmap="inferno",
                      clim=[tmin, tmax], point_size=6, render_points_as_spheres=True)
    plotter.add_text(f"Predicted (t={physical_time:.2f}s)", font_size=10)

    cloud_err = pv.PolyData(pos_np)
    cloud_err["AbsError"] = abs_err_t
    plotter.subplot(0, 2)
    plotter.add_mesh(cloud_err, scalars="AbsError", cmap="viridis",
                      point_size=6, render_points_as_spheres=True)
    plotter.add_text(f"Abs Error (max={abs_err_t.max():.1f}K)", font_size=10)

    plotter.link_views()
    plotter.screenshot(out_path)
    plotter.close()
    print(f"Saved spatial comparison -> {out_path}")


def plot_temporal_curve(data, pred, true, out_path):
    time_s = data.time_stamps.numpy()
    true_curve = true.mean(dim=0).numpy()
    pred_curve = pred.mean(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_s, true_curve, "o-", label="Actual (mean over nodes)", color="black")
    ax.plot(time_s, pred_curve, "s--", label="Predicted (mean over nodes)", color="crimson")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Temperature (K)")
    ax.set_title(f"{data.sim_id} -- Temperature evolution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved temporal curve -> {out_path}")


def plot_error_histogram(pred, true, t_idx, out_path):
    abs_err_t = (pred[:, t_idx] - true[:, t_idx]).abs().numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(abs_err_t, bins=50, color="steelblue", edgecolor="black")
    ax.axvline(np.percentile(abs_err_t, 95), color="orange", linestyle="--",
               label=f"95th pct = {np.percentile(abs_err_t, 95):.2f}K")
    ax.axvline(abs_err_t.max(), color="red", linestyle="--",
               label=f"max = {abs_err_t.max():.2f}K")
    ax.set_xlabel("Absolute Error (K)")
    ax.set_ylabel("Node count")
    ax.set_title("Per-node absolute error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved error histogram -> {out_path}")


def plot_pred_vs_true_scatter(pred, true, out_path):
    true_flat = true.flatten().numpy()
    pred_flat = pred.flatten().numpy()

    max_points = 20000
    if len(true_flat) > max_points:
        idx = np.random.choice(len(true_flat), max_points, replace=False)
        true_flat, pred_flat = true_flat[idx], pred_flat[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_flat, pred_flat, s=3, alpha=0.3, color="teal")
    lims = [min(true_flat.min(), pred_flat.min()), max(true_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, "r--", label="y = x (perfect prediction)")
    ax.set_xlabel("Actual Temperature (K)")
    ax.set_ylabel("Predicted Temperature (K)")
    ax.set_title("Predicted vs. Actual (all nodes, all timesteps)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved pred-vs-true scatter -> {out_path}")


def main():
    model, stats = load_model_and_stats()
    test_list = torch.load(os.path.join(SPLITS_DIR, "test.pt"), weights_only=False)

    if VARIANT_SIM_IDS is not None:
        chosen = [d for d in test_list if d.sim_id in VARIANT_SIM_IDS]
    else:
        chosen = test_list[:N_VARIANTS_TO_PLOT]

    print(f"Visualizing {len(chosen)} variant(s): {[d.sim_id for d in chosen]}")

    for data in chosen:
        pred, true = run_inference(model, stats, data)
        n_steps = true.shape[1]
        t_idx = TIMESTEP_INDEX_FOR_SPATIAL_PLOTS
        if t_idx < 0:
            t_idx = n_steps + t_idx

        variant_dir = os.path.join(VIZ_DIR, data.sim_id)
        os.makedirs(variant_dir, exist_ok=True)

        plot_spatial_comparison(data, pred, true, t_idx,
            os.path.join(variant_dir, f"spatial_comparison_t{t_idx}.png"))
        plot_temporal_curve(data, pred, true,
            os.path.join(variant_dir, "temporal_curve.png"))
        plot_error_histogram(pred, true, t_idx,
            os.path.join(variant_dir, f"error_histogram_t{t_idx}.png"))
        plot_pred_vs_true_scatter(pred, true,
            os.path.join(variant_dir, "pred_vs_true_scatter.png"))

    print(f"\nAll visualizations saved under: {VIZ_DIR}")


if __name__ == "__main__":
    main()