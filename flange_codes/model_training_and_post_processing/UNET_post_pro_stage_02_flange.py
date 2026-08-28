"""
STAGE 2: Render target / predicted / error temperature fields on the actual
3D mesh (using one original .vtu for correct topology), plus error-over-time
and probe-point plots, for a single chosen Brake variant.

Run this once per variant (BEST / MEDIAN / WORST) by changing INPUT_NPZ
and SAMPLE_VTU below.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os

# ---- EDIT THESE per run ----
#INPUT_NPZ   = r"E:\NeurIPS_dataset\flange_models\field_data_flange_UNET\field_data_Flange_p973_m353_v05.npz"   # BEST
#INPUT_NPZ   = r"E:\NeurIPS_dataset\flange_models\field_data_flange_UNET\field_data_Flange_p1033_m353_v06.npz"   # Median 
INPUT_NPZ   = r"E:\NeurIPS_dataset\flange_models\field_data_flange_UNET\field_data_Flange_p1033_m403_v07.npz"   # Worst

#SAMPLE_VTU  = r"D:\NeurIPS\surrogate_flange\results_case_973\case_973_353\variant_05\results\flange_pour973_mould353_data_t0001.vtu" # BEst 
#SAMPLE_VTU  = r"D:\NeurIPS\surrogate_flange\results_case_1033\case_1033_353\variant_06\results\flange_pour1033_mould353_data_t0001.vtu" # Median 
SAMPLE_VTU  = r"D:\NeurIPS\surrogate_flange\results_case_1033\case_1033_403\variant_07\results\flange_pour1033_mould403_data_t0001.vtu" # Worst 
# -----------------------------
#OUT_DIR = r"E:\NeurIPS_dataset\flange_models\field_plots\UNET_filed_plots\best_Flange_p973_m353_v05" # best
#OUT_DIR = r"E:\NeurIPS_dataset\flange_models\field_plots\UNET_filed_plots\median_Flange_p1033_m353_v06" # median
OUT_DIR = r"E:\NeurIPS_dataset\flange_models\field_plots\UNET_filed_plots\worst_Flange_p1033_m403_v07" # worst

os.makedirs(OUT_DIR, exist_ok=True)

data = np.load(INPUT_NPZ, allow_pickle=True)
times = data["times"]
gt_temp = data["gt_temp"]
pred_temp = data["pred_temp"]
abs_err = data["abs_err"]
sim_id = str(data["sim_id"])
geometry = str(data["geometry"])

mesh = pv.read(SAMPLE_VTU)  # topology only; point_data overwritten below

# Guard against mesh/field mismatches (wrong variant's .vtu picked by mistake)
if mesh.n_points != gt_temp.shape[0]:
    raise ValueError(
        f"Mesh/field mismatch: SAMPLE_VTU has {mesh.n_points} points but "
        f"the .npz ground truth has {gt_temp.shape[0]} nodes. "
        f"Make sure SAMPLE_VTU is from the SAME variant folder as INPUT_NPZ "
        f"(check pour/mould/variant numbers match)."
    )


def snapshot_indices(n_t, k=4):
    return np.linspace(0, n_t - 1, k, dtype=int)


# --------------------------------------------------------------------------
# Fig A: Target vs Predicted vs Error, rendered on the real 3D mesh
# --------------------------------------------------------------------------
def fig_target_vs_predicted_3d():
    idxs = snapshot_indices(len(times), k=4)
    vmin = min(gt_temp.min(), pred_temp.min())
    vmax = max(gt_temp.max(), pred_temp.max())

    # FIX: one shared error color scale across all shown snapshots in THIS
    # figure, computed from the actual data at those snapshots. Previously
    # each error panel auto-scaled independently but only the last panel's
    # scale bar was shown -- meaning every panel looked like it was on the
    # same scale when it wasn't (e.g. an early-transient panel with ~50K
    # error could render the same shade of "hot yellow" as a late-time
    # panel with ~5K error). This makes the colors mean what they say.
    err_vmin = 0.0
    err_vmax = float(abs_err[:, idxs].max())

    plotter = pv.Plotter(shape=(3, len(idxs)), off_screen=True, window_size=(1400, 1000))

    for col, ti in enumerate(idxs):
        m = mesh.copy()
        m.point_data["Target"] = gt_temp[:, ti]
        m.point_data["Predicted"] = pred_temp[:, ti]
        m.point_data["AbsError"] = abs_err[:, ti]

        plotter.subplot(0, col)
        plotter.add_mesh(m, scalars="Target", cmap="inferno", clim=[vmin, vmax],
                          show_scalar_bar=(col == len(idxs) - 1),
                          scalar_bar_args={"title": "Temp (K)"})
        plotter.add_text(f"Target  t={times[ti]:.1f}s", font_size=9)

        plotter.subplot(1, col)
        plotter.add_mesh(m, scalars="Predicted", cmap="inferno", clim=[vmin, vmax],
                          show_scalar_bar=(col == len(idxs) - 1),
                          scalar_bar_args={"title": "Temp (K)"})
        plotter.add_text(f"Predicted  t={times[ti]:.1f}s", font_size=9)

        plotter.subplot(2, col)
        plotter.add_mesh(m, scalars="AbsError", cmap="viridis",
                          clim=[err_vmin, err_vmax],
                          show_scalar_bar=(col == len(idxs) - 1),
                          scalar_bar_args={"title": "|Error| (K)"})
        plotter.add_text(f"|Error|  t={times[ti]:.1f}s", font_size=9)

    plotter.link_views()
    out = os.path.join(OUT_DIR, f"fieldmap3d_{geometry}_{sim_id}.png")
    plotter.screenshot(out)
    plotter.close()
    print(f"Saved {out}")
    print(f"  Error color scale used for all 4 panels: [{err_vmin:.2f}, {err_vmax:.2f}] K")


# --------------------------------------------------------------------------
# Fig B: Error over time
# --------------------------------------------------------------------------
def fig_error_over_time():
    mean_err_t = abs_err.mean(axis=0)
    max_err_t = abs_err.max(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, mean_err_t, label="Mean Abs Error", color="#2c7fb8")
    ax.plot(times, max_err_t, label="Max Abs Error", color="#e6550d")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute Error (K)")
    ax.set_title(f"Prediction Error Over Time — {geometry} / {sim_id}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(OUT_DIR, f"error_over_time_{geometry}_{sim_id}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------------------------------------------
# Fig C: Probe-point time series (hottest, coldest, mid-range nodes)
# --------------------------------------------------------------------------
def fig_probe_points():
    final_gt = gt_temp[:, -1]
    hottest = int(np.argmax(final_gt))
    coldest = int(np.argmin(final_gt))
    mid = int(np.argsort(final_gt)[len(final_gt) // 2])
    probes = {"Hottest node": hottest, "Coldest node": coldest, "Mid node": mid}

    fig, axes = plt.subplots(1, len(probes), figsize=(4.5 * len(probes), 4), sharey=True)
    for ax, (label, pid) in zip(axes, probes.items()):
        ax.plot(times, gt_temp[pid], label="Target", color="black", linewidth=1.5)
        ax.plot(times, pred_temp[pid], label="Predicted", color="#e6550d", linestyle="--")
        ax.set_title(f"{label} (id={pid})")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Temperature (K)")
    axes[0].legend()
    fig.suptitle(f"Probe-Point Temperature Traces — {geometry} / {sim_id}", fontweight="bold")
    out = os.path.join(OUT_DIR, f"probe_points_{geometry}_{sim_id}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_target_vs_predicted_3d()
    fig_error_over_time()
    fig_probe_points()
    print("\nAll field plots saved to", OUT_DIR)