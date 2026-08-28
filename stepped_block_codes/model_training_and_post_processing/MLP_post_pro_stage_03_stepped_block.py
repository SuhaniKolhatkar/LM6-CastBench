"""
STAGE 3: Combined BEST / MEDIAN / WORST comparison figure for Brake, with
a SINGLE shared error color scale across all three variants -- so a reader
can directly compare severity by eye, not just within one variant's own
snapshots (which is what the individual Stage 2 figures already give you).

Produces:
  fieldmap3d_comparison_stepped_block.png   -- 3 rows (Best/Median/Worst) x 4 time
                                        columns, Error field only, one
                                        shared color scale across all 12 panels
  error_over_time_comparison_stepped_block.png -- one plot, 3 variants overlaid

Run once after all three Stage 1 .npz files exist for Flnage.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os

# ---- EDIT THESE ----
VARIANTS = [
    {
        "label": "Best",
        "npz":r"E:\NeurIPS_dataset\stepped_block\field_data_stepped_block_chkpt_MLP\field_data_Stepped_Block_p1033_m303_v07.npz",
        "vtu": r"D:\NeurIPS\surrogate_block\results_case_1033\case_1033_303\variant_07\results\block_data_t0001.vtu",
    },
    {
        "label": "Median",
        "npz": r"E:\NeurIPS_dataset\stepped_block\field_data_stepped_block_chkpt_MLP\field_data_Stepped_Block_p993_m353_v02.npz",
        "vtu": r"D:\NeurIPS\surrogate_block\results_case_993\case_993_353\variant_02\results\block_data_t0001.vtu",
    },
    {
        "label": "Worst",
        "npz": r"E:\NeurIPS_dataset\stepped_block\field_data_stepped_block_chkpt_MLP\field_data_Stepped_Block_p953_m403_v03.npz",
        "vtu":  r"D:\NeurIPS\surrogate_block\results_case_953\case_953_403\variant_03\results\block_data_t0001.vtu",
    },
]
GEOMETRY_NAME = "stepped_block"
OUT_DIR = "field_plots/MLP_field_plots/comparison"
# ---------------------

os.makedirs(OUT_DIR, exist_ok=True)


def snapshot_indices(n_t, k=4):
    return np.linspace(0, n_t - 1, k, dtype=int)


# --------------------------------------------------------------------------
# Load all three variants up front, and validate each mesh matches its field
# --------------------------------------------------------------------------
loaded = []
for v in VARIANTS:
    data = np.load(v["npz"], allow_pickle=True)
    mesh = pv.read(v["vtu"])

    if mesh.n_points != data["gt_temp"].shape[0]:
        raise ValueError(
            f"[{v['label']}] Mesh/field mismatch: vtu has {mesh.n_points} points "
            f"but npz has {data['gt_temp'].shape[0]} nodes. Check the vtu path "
            f"matches this variant exactly."
        )

    loaded.append({
        "label": v["label"],
        "mesh": mesh,
        "times": data["times"],
        "gt_temp": data["gt_temp"],
        "pred_temp": data["pred_temp"],
        "abs_err": data["abs_err"],
        "sim_id": str(data["sim_id"]),
    })

print("Loaded variants:", [d["sim_id"] for d in loaded])


# --------------------------------------------------------------------------
# Fig 1: Combined field-map comparison -- 3 rows x 4 time columns, Error
# only, ONE shared color scale computed across ALL 12 panels
# --------------------------------------------------------------------------
def fig_comparison_fieldmap():
    n_rows = len(loaded)
    n_cols = 4

    # Shared color scale, computed honestly from every panel that will
    # actually be shown (not the global all-time max of any single variant,
    # and not a number chosen after looking at the plots)
    all_shown_errs = []
    per_variant_idxs = []
    for d in loaded:
        idxs = snapshot_indices(len(d["times"]), k=n_cols)
        per_variant_idxs.append(idxs)
        all_shown_errs.append(d["abs_err"][:, idxs])
    err_vmin = 0.0
    err_vmax = float(max(a.max() for a in all_shown_errs))

    plotter = pv.Plotter(shape=(n_rows, n_cols), off_screen=True,
                          window_size=(1800, 200 * n_rows + 100))

    for row, (d, idxs) in enumerate(zip(loaded, per_variant_idxs)):
        for col, ti in enumerate(idxs):
            m = d["mesh"].copy()
            m.point_data["AbsError"] = d["abs_err"][:, ti]

            plotter.subplot(row, col)
            plotter.add_mesh(
                m, scalars="AbsError", cmap="viridis",
                clim=[err_vmin, err_vmax],
                show_scalar_bar=(row == n_rows - 1 and col == n_cols - 1),
                scalar_bar_args={"title": "|Error| (K)", "fmt": "%.1f"},
            )
            plotter.add_text(
                f"{d['label']}  t={d['times'][ti]:.1f}s", font_size=9
            )

    plotter.link_views()
    out = os.path.join(OUT_DIR, f"fieldmap3d_comparison_{GEOMETRY_NAME}.png")
    plotter.screenshot(out)
    plotter.close()
    print(f"Saved {out}")
    print(f"  Shared error color scale (all 12 panels): [{err_vmin:.2f}, {err_vmax:.2f}] K")


# --------------------------------------------------------------------------
# Fig 2: Combined error-over-time -- one plot, 3 variants overlaid
# --------------------------------------------------------------------------
def fig_comparison_error_over_time():
    colors = {"Best": "#31a354", "Median": "#e6e20d", "Worst": "#de2d26"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for d in loaded:
        c = colors.get(d["label"], None)
        mean_err_t = d["abs_err"].mean(axis=0)
        max_err_t = d["abs_err"].max(axis=0)
        axes[0].plot(d["times"], mean_err_t, label=d["label"], color=c)
        axes[1].plot(d["times"], max_err_t, label=d["label"], color=c)

    axes[0].set_title("Mean Absolute Error Over Time", fontweight="bold")
    axes[1].set_title("Max Absolute Error Over Time", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absolute Error (K)")
        ax.legend(title="Variant")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Error Comparison Across Variants — {GEOMETRY_NAME}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"error_over_time_comparison_{GEOMETRY_NAME}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_comparison_fieldmap()
    fig_comparison_error_over_time()
    print(f"\nAll comparison figures saved to {OUT_DIR}")