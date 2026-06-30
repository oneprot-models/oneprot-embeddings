"""
plot_average_roc.py

Single figure with 7 subplots (one per model), each showing 16 average ROC
curves averaged over seeds 11–15.  Curves from the same dataset family share
a colour palette:

    ASD_merged_*   →  reds / oranges   (4 curves)
    ASD_pockets_*  →  blues            (4 curves)
    merged_*       →  greens           (4 curves)
    Kinase_*       →  purples          (4 curves)

Matches the flat file naming scheme from evaluate.py (document 4):
    <base_dir>/{task}_{model}_seed{seed}_preds.npz

Usage:
    python plot_average_roc.py \
        --base_dir <REPO_ROOT>/curve_plots \
        --output_dir <REPO_ROOT>/average_roc_plots
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Short display names for subplot titles
MODEL_LABELS = {
    "oneprot_pocket_text_32900":                           "Pocket_Text",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "SG+ST+Pocket+Text",
    "oneprot_md_combined_gpcr_no_struct_graph_32900":      "MD+ST+Pocket+Text",
    "oneprot_md_combined_gpcr_no_struct_token_32900":      "MD+SG+Pocket+Text",
    "oneprot_struct_graph_pocket_text_32900":              "SG+Pocket+Text",
    "oneprot_md_combined_gpcr_32900":                      "MD+SG+ST+Pocket+Text",
    "oneprot_struct_token_pocket_text_32900":              "ST+Pocket+Text",
}

TASKS = [
    # ASD_merged family  (reds/oranges)
    "ASD_merged_pocket_binary_comp",
    "ASD_merged_pocket_binary_text_comp",
    "ASD_merged_pocket_sequence_binary_comp",
    "ASD_merged_pocket_sequence_binary_text_comp",
    # ASD_pockets family  (blues)
    "ASD_pockets_binary_comp",
    "ASD_pockets_binary_text_comp",
    "ASD_pockets_sequence_binary_comp",
    "ASD_pockets_sequence_binary_text_comp",
    # merged family  (greens)
    "merged_pocket_binary_comp",
    "merged_pocket_binary_text_comp",
    "merged_pocket_sequence_binary_comp",
    "merged_pocket_sequence_binary_text_comp",
    # Kinase family  (purples)
    "Kinase_combined",
    "Kinase_combined_text",
    "Kinase_pocket",
    "Kinase_pocket_text",
]

# ── Colour palettes grouped by dataset family ─────────────────────────────────
# 4 shades per family, going light → dark so individual curves stay distinct
FAMILY_COLORS = {
    "ASD_merged": ["#ff9999", "#ff4444", "#cc0000", "#800000"],   # reds
    "ASD_pockets": ["#99b3ff", "#4477ff", "#0033cc", "#001a66"],  # blues
    "merged":     ["#99dd99", "#44bb44", "#228822", "#0d4d0d"],   # greens
    "Kinase":     ["#cc99ff", "#9933ff", "#6600cc", "#330066"],   # purples
}

FAMILY_PATCH_COLORS = {
    "ASD_merged":  "#cc0000",
    "ASD_pockets": "#0033cc",
    "merged":      "#228822",
    "Kinase":      "#6600cc",
}

def task_family(task: str) -> str:
    if task.startswith("ASD_merged"):
        return "ASD_merged"
    elif task.startswith("ASD_pockets"):
        return "ASD_pockets"
    elif task.startswith("merged"):
        return "merged"
    elif task.startswith("Kinase"):
        return "Kinase"
    return "other"

# Pre-assign a color to each task based on its position within its family
TASK_COLORS = {}
family_counters = {"ASD_merged": 0, "ASD_pockets": 0, "merged": 0, "Kinase": 0}
for task in TASKS:
    fam = task_family(task)
    idx = family_counters[fam]
    TASK_COLORS[task] = FAMILY_COLORS[fam][idx % 4]
    family_counters[fam] += 1

# Short task labels (strip common suffixes for readability)
def short_label(task: str) -> str:
    return (task
            .replace("ASD_merged_pocket_", "ASDm_p_")
            .replace("ASD_pockets_", "PL8p_")
            .replace("merged_pocket_", "PL8_Kin_")
            .replace("_binary_comp", "")
            .replace("_binary_text_comp", "_t")
            .replace("_sequence_binary_comp", "_s")
            .replace("_sequence_binary_text_comp", "_st")
            .replace("_combined", "_sp")
            .replace("_pocket", "_p")
            .replace("_text", "_txt"))

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 200)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_seed_curve(base_dir: str, model: str, task: str, seed: int):
    fname = f"{task}_{model}_seed{seed}_preds.npz"
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr, sklearn_auc(fpr, tpr)


def average_roc(base_dir: str, model: str, task: str):
    tprs, aucs = [], []
    for seed in SEEDS:
        result = load_seed_curve(base_dir, model, task, seed)
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


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(base_dir: str, output_dir: str):
    n_models = len(MODELS)
    ncols = 4
    nrows = 2  # 7 subplots → 4+3, last cell left empty

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 5.0),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    # Hide the unused 8th cell
    axes_flat[-1].set_visible(False)

    for ax_idx, model in enumerate(MODELS):
        ax = axes_flat[ax_idx]
        missing = []

        for task in TASKS:
            result = average_roc(base_dir, model, task)
            if result is None:
                missing.append(task)
                continue

            mean_tpr, std_tpr, mean_auc, std_auc, n = result
            color = TASK_COLORS[task]
            n_label = f"(n={n}) " if n < len(SEEDS) else ""
            label = f"{n_label}{short_label(task)}  {mean_auc:.2f}±{std_auc:.2f}"

            ax.plot(MEAN_FPR, mean_tpr, color=color, lw=1.4, label=label)
            ax.fill_between(
                MEAN_FPR,
                mean_tpr - std_tpr,
                mean_tpr + std_tpr,
                color=color,
                alpha=0.12,
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=5.5, loc="lower right", framealpha=0.8, ncol=1)

        if missing:
            print(f"  [{MODEL_LABELS[model]}] Missing: {missing}")

    # ── Shared family-colour legend below the figure ──────────────────────────
    family_patches = [
        mpatches.Patch(color=FAMILY_PATCH_COLORS[fam], label=fam)
        for fam in ["ASD_merged", "ASD_pockets", "merged", "Kinase"]
    ]
    fig.legend(
        handles=family_patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        title="Dataset family",
        title_fontsize=10,
        bbox_to_anchor=(0.5, -0.03),
        framealpha=0.9,
    )

    fig.suptitle(
        f"Average ROC curves (seeds {SEEDS[0]}–{SEEDS[-1]})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "all_models_avg_roc.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


def list_available_files(base_dir: str):
    files = [f for f in os.listdir(base_dir) if f.endswith("_preds.npz")]
    print(f"Found {len(files)} .npz files in {base_dir}:")
    for f in sorted(files)[:20]:
        print(f"  {f}")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="<REPO_ROOT>/curve_plots",
    )
    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/average_roc_plots",
    )
    parser.add_argument(
        "--list_files", action="store_true",
        help="Print available .npz files and exit (for debugging)",
    )
    args = parser.parse_args()

    if args.list_files:
        list_available_files(args.base_dir)
        return

    print(f"Reading from : {args.base_dir}")
    print(f"Writing to   : {args.output_dir}")
    plot_all(args.base_dir, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()