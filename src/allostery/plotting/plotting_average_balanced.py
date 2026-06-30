"""
plot_balanced_vs_unbalanced_roc.py

Single figure with 7 subplots (one per model).  Each subplot shows 8 average
ROC curves:
  - 4 ASD_merged tasks from the UNBALANCED run  (flat file naming, reds, solid)
  - 4 ASD_merged tasks from the BALANCED run    (hierarchical folder naming, blues, dashed)

Unbalanced files (flat):
    <unbalanced_dir>/{task}_{model}_seed{seed}_preds.npz

Balanced files (hierarchical, with _balanced suffix):
    <balanced_dir>/<model>/{task}_balanced/seed{seed}/preds.npz

Usage:
    python plot_balanced_vs_unbalanced_roc.py \
        --unbalanced_dir <REPO_ROOT>/curve_plots \
        --balanced_dir   <CURVE_PLOTS_ROOT> \
        --output_dir     <REPO_ROOT>/balanced_vs_unbalanced_roc
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc as sklearn_auc

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS = [
    "oneprot_pocket_text_32900",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity",
    "oneprot_md_combined_gpcr_no_struct_graph_32900",
    "oneprot_md_combined_gpcr_no_struct_token_32900",
    "oneprot_struct_graph_pocket_text_32900",
    "oneprot_md_combined_gpcr_32900",
    "oneprot_struct_token_pocket_text_32900",
]

MODEL_LABELS = {
    "oneprot_pocket_text_32900":                               "pocket_text",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "full_allatom",
    "oneprot_md_combined_gpcr_no_struct_graph_32900":          "md_no_graph",
    "oneprot_md_combined_gpcr_no_struct_token_32900":          "md_no_token",
    "oneprot_struct_graph_pocket_text_32900":                  "struct_graph",
    "oneprot_md_combined_gpcr_32900":                          "md_combined",
    "oneprot_struct_token_pocket_text_32900":                  "struct_token",
}

TASKS = [
    "ASD_merged_pocket_binary_comp",
    "ASD_merged_pocket_binary_text_comp",
    "ASD_merged_pocket_sequence_binary_comp",
    "ASD_merged_pocket_sequence_binary_text_comp",
]

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 200)

# Unbalanced → reds (light to dark), solid lines
UNBALANCED_COLORS = ["#ffaaaa", "#ff5555", "#cc0000", "#7a0000"]
# Balanced   → blues (light to dark), dashed lines
BALANCED_COLORS   = ["#aac4ff", "#4477ff", "#0033cc", "#001166"]

def short_label(task: str) -> str:
    return (task
            .replace("ASD_merged_pocket_", "")
            .replace("_binary_comp", "_bc")
            .replace("_binary_text_comp", "_btc")
            .replace("_sequence_binary_comp", "_sbc")
            .replace("_sequence_binary_text_comp", "_sbtc"))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_flat(base_dir, model, task, seed):
    path = os.path.join(base_dir, f"{task}_{model}_seed{seed}_preds.npz")
    return _load_npz(path)

def load_hierarchical(base_dir, model, task, seed):
    path = os.path.join(base_dir, model, f"{task}_balanced", f"seed{seed}", "preds.npz")
    return _load_npz(path)

def _load_npz(path):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    y_true, y_pred = data["y_true"], data["y_pred"]
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr, sklearn_auc(fpr, tpr)

def average_roc(load_fn, base_dir, model, task):
    tprs, aucs = [], []
    for seed in SEEDS:
        result = load_fn(base_dir, model, task, seed)
        if result is None:
            continue
        fpr, tpr, roc_auc = result
        tprs.append(np.interp(MEAN_FPR, fpr, tpr))
        aucs.append(roc_auc)
    if not tprs:
        return None
    return (
        np.mean(tprs, axis=0),
        np.std(tprs, axis=0),
        float(np.mean(aucs)),
        float(np.std(aucs)),
        len(tprs),
    )


# ── Per-subplot legend: two clearly labelled sections ─────────────────────────

def make_subplot_legend(ax):
    """
    Two-section legend inside each subplot:
      Section 1 header: '— Unbalanced' (red swatch)
      Section 2 header: '-- Balanced'  (blue swatch)
      Then 4 task-shade entries each.
    """
    handles = []

    # Section header: Unbalanced
    handles.append(mlines.Line2D(
        [], [], color="white", label="── UNBALANCED ──",
        linewidth=0,
    ))
    for i, task in enumerate(TASKS):
        handles.append(mlines.Line2D(
            [], [], color=UNBALANCED_COLORS[i], lw=2.0, linestyle="-",
            label=short_label(task),
        ))

    # Section header: Balanced
    handles.append(mlines.Line2D(
        [], [], color="white", label="── BALANCED ──",
        linewidth=0,
    ))
    for i, task in enumerate(TASKS):
        handles.append(mlines.Line2D(
            [], [], color=BALANCED_COLORS[i], lw=2.0, linestyle="--",
            label=short_label(task),
        ))

    legend = ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=5.2,
        framealpha=0.88,
        ncol=1,
        handlelength=2.0,
    )
    # Bold the two section headers
    for text in legend.get_texts():
        if "UNBALANCED" in text.get_text() or "BALANCED" in text.get_text():
            text.set_fontweight("bold")
            text.set_color("#333333")


# ── Main subplot drawing ───────────────────────────────────────────────────────

def plot_subplot(ax, model, unbalanced_dir, balanced_dir):
    missing = []

    for i, task in enumerate(TASKS):
        lbl = short_label(task)

        # Unbalanced — solid red shades
        res_u = average_roc(load_flat, unbalanced_dir, model, task)
        if res_u is None:
            missing.append(f"unbal/{task}")
        else:
            mean_tpr, std_tpr, mean_auc, std_auc, n = res_u
            col = UNBALANCED_COLORS[i]
            ax.plot(MEAN_FPR, mean_tpr, color=col, lw=1.6, linestyle="-")
            ax.fill_between(MEAN_FPR, mean_tpr - std_tpr, mean_tpr + std_tpr,
                            color=col, alpha=0.12)

        # Balanced — dashed blue shades
        res_b = average_roc(load_hierarchical, balanced_dir, model, task)
        if res_b is None:
            missing.append(f"bal/{task}")
        else:
            mean_tpr, std_tpr, mean_auc, std_auc, n = res_b
            col = BALANCED_COLORS[i]
            ax.plot(MEAN_FPR, mean_tpr, color=col, lw=1.6, linestyle="--")
            ax.fill_between(MEAN_FPR, mean_tpr - std_tpr, mean_tpr + std_tpr,
                            color=col, alpha=0.12)

    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    make_subplot_legend(ax)

    if missing:
        print(f"  [{MODEL_LABELS[model]}] Missing: {missing}")


# ── Figure assembly ───────────────────────────────────────────────────────────

def plot_all(unbalanced_dir, balanced_dir, output_dir):
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 5.4),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()
    axes_flat[-1].set_visible(False)

    for ax_idx, model in enumerate(MODELS):
        plot_subplot(axes_flat[ax_idx], model, unbalanced_dir, balanced_dir)

    # ── Figure-level colour key (bottom centre) ───────────────────────────────
    shade_handles = []
    for i, task in enumerate(TASKS):
        lbl = short_label(task)
        shade_handles.append(mpatches.Patch(
            color=UNBALANCED_COLORS[i],
            label=f"{lbl}  (red=unbal, blue=bal)",
        ))

    style_handles = [
        mlines.Line2D([], [], color="#333333", lw=2, linestyle="-",
                      label="Solid  =  Unbalanced split"),
        mlines.Line2D([], [], color="#333333", lw=2, linestyle="--",
                      label="Dashed =  Balanced split"),
    ]

    fig.legend(
        handles=style_handles + shade_handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.07),
        framealpha=0.9,
        title="Line style encodes split type  |  Shade index encodes task variant",
        title_fontsize=9,
    )

    fig.suptitle(
        f"ASD_merged tasks  —  Balanced vs Unbalanced training split  "
        f"(seeds {SEEDS[0]}–{SEEDS[-1]})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "balanced_vs_unbalanced_avg_roc.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unbalanced_dir",
        default="<REPO_ROOT>/curve_plots",
    )
    parser.add_argument(
        "--balanced_dir",
        default="<CURVE_PLOTS_ROOT>",
    )
    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/balanced_vs_unbalanced_roc",
    )
    args = parser.parse_args()

    print(f"Unbalanced dir : {args.unbalanced_dir}")
    print(f"Balanced dir   : {args.balanced_dir}")
    print(f"Output dir     : {args.output_dir}")
    plot_all(args.unbalanced_dir, args.balanced_dir, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()