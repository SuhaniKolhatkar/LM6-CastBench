"""
visualize_mlp_brake.py

WHAT THIS DOES
---------------
Loads the trained Brake MLP model + its saved normalization stats,
picks one (or more) test variants, runs inference, and produces four
kinds of visualizations:

  1. 3D spatial field plots (PyVista): actual temperature, predicted
     temperature, and absolute error, all rendered on the real node
     positions of the casting geometry, at a chosen timestep.
  2. Time-evolution curve: mean temperature vs. time, actual vs.
     predicted (visualizes C5 -- does the model track the cooling
     history correctly, or drift over time?).
  3. Error histogram: distribution of per-node absolute errors at the
     chosen timestep (visualizes C4 -- worst-case/local error).
  4. Predicted-vs-true scatter plot: a standard regression sanity
     check -- points should lie on the y=x diagonal if predictions
     are accurate.

All PyVista renders are saved as PNG screenshots (off-screen, so this
runs even without a display attached), and all matplotlib plots are
saved as PNG too. Nothing is required to already be open on screen.

Requires: pip install pyvista matplotlib numpy torch
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyvista as pv

from mlp_model import TemperatureMLP, denormalize_targets

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
GEOMETRY_FAMILY = "Brake"
SPLITS_DIR = r"E:\NeurIPS_dataset\brake_models\brake_splits"
RESULTS_DIR = r"E:\NeurIPS_dataset\brake_models\brake_results\mlp"
VIZ_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_DIR, "mlp_brake.pt")
STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats_brake.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which test variant(s) to visualize. None = visualize the first N found.
VARIANT_SIM_IDS = None      # e.g. ["Brake_p953_m303_v01"] to pick specific ones
N_VARIANTS_TO_PLOT = 2      # used only if VARIANT_SIM_IDS is None

# Which timestep index to use for the 3D spatial plots (0 = first frame).
# Set to -1 to use the LAST timestep instead (fully cooled state).
TIMESTEP_INDEX_FOR_SPATIAL_PLOTS = -1


# ---------------------------------------------------------------------
# Load model, stats, test data
# ---------------------------------------------------------------------
def load_model_and_stats():
    stats = torch.load(STATS_PATH, weights_only=False)
    model = TemperatureMLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, stats


def run_inference(model, stats, data):
    """Returns pred [N, T] and true [N, T] temperature tensors (Kelvin)."""
    n_nodes, n_steps = data.x.shape[0], data.y.shape[1]
    static_rep = data.x.unsqueeze(1).expand(n_nodes, n_steps, 6)
    time_rep = data.time_stamps.view(1, n_steps, 1).expand(n_nodes, n_steps, 1)
    inputs = torch.cat([static_rep, time_rep], dim=2).reshape(n_nodes * n_steps, 7)
    inputs_n = ((inputs - stats["input_mean"]) / stats["input_std"]).to(DEVICE)

    with torch.no_grad():
        pred_n = model(inputs_n)
    pred = denormalize_targets(pred_n.cpu(), stats).reshape(n_nodes, n_steps)
    true = data.y
    return pred, true


# ---------------------------------------------------------------------
# 1. 3D spatial field comparison (PyVista) -- actual / predicted / error
# ---------------------------------------------------------------------
def plot_spatial_comparison(data, pred, true, t_idx, out_path):
    pos_np = data.pos.numpy()          # [N, 3] meters
    true_t = true[:, t_idx].numpy()    # [N]
    pred_t = pred[:, t_idx].numpy()    # [N]
    abs_err_t = np.abs(pred_t - true_t)

    physical_time = float(data.time_stamps[t_idx])

    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1800, 600))

    # Shared color scale for actual vs predicted so they're visually comparable
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


# ---------------------------------------------------------------------
# 2. Time-evolution curve: mean temperature vs time, actual vs predicted
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 3. Error histogram at the chosen timestep
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 4. Predicted vs. true scatter plot (regression sanity check)
# ---------------------------------------------------------------------
def plot_pred_vs_true_scatter(pred, true, out_path):
    true_flat = true.flatten().numpy()
    pred_flat = pred.flatten().numpy()

    # Subsample if huge, so the plot isn't a solid blob and renders fast
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


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
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
            t_idx = n_steps + t_idx  # convert -1 -> last index

        variant_dir = os.path.join(VIZ_DIR, data.sim_id)
        os.makedirs(variant_dir, exist_ok=True)

        plot_spatial_comparison(
            data, pred, true, t_idx,
            os.path.join(variant_dir, f"spatial_comparison_t{t_idx}.png")
        )
        plot_temporal_curve(
            data, pred, true,
            os.path.join(variant_dir, "temporal_curve.png")
        )
        plot_error_histogram(
            pred, true, t_idx,
            os.path.join(variant_dir, f"error_histogram_t{t_idx}.png")
        )
        plot_pred_vs_true_scatter(
            pred, true,
            os.path.join(variant_dir, "pred_vs_true_scatter.png")
        )

    print(f"\nAll visualizations saved under: {VIZ_DIR}")


if __name__ == "__main__":
    main()