"""
plot_figure1_representative_roc.py

Figure 1: representative ROC comparison for a single model across datasets.

Design:
    - fixed model: md_no_graph
    - 4 subplots = dataset regimes
        1) PL8
        2) Kinase
        3) PL8 + Kinase
        4) ASD + PL8 + Kinase   (BALANCED ASD run)
    - 4 curves per subplot = embedding sets
        pocket
        pocket+text
        pocket+sequence
        pocket+sequence+text

Data sources:
    - PL8, merged, Kinase -> flat naming from evaluate.py
          <base_dir>/{task}_{model}_seed{seed}_preds.npz
    - BALANCED ASD_merged -> hierarchical naming
          <balanced_dir>/<model>/{task}_balanced/seed{seed}/preds.npz

Usage:
    python plot_figure1_representative_roc.py \
        --base_dir <REPO_ROOT>/curve_plots \
        --balanced_dir <CURVE_PLOTS_ROOT> \
        --output_dir <REPO_ROOT>/figure1_representative
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "oneprot_md_combined_gpcr_no_struct_graph_32900"
MODEL_LABEL = "MD+ST+Pocket+Text"

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 200)

# Canonical embedding order used in every subplot
EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]
EMBEDDING_LABELS = {
    "p": "pocket",
    "pt": "pocket+text",
    "ps": "pocket+sequence",
    "pst": "pocket+sequence+text",
}

# Keep colors consistent across all subplots
EMBEDDING_COLORS = {
    "p":   "#1f77b4",  # blue
    "pt":  "#ff7f0e",  # orange
    "ps":  "#2ca02c",  # green
    "pst": "#d62728",  # red
}

# 4 dataset panels
DATASETS = [
    {
        "panel_title": "PL8",
        "source": "flat",
        "tasks": {
            "p":   "ASD_pockets_binary_comp",
            "pt":  "ASD_pockets_binary_text_comp",
            "ps":  "ASD_pockets_sequence_binary_comp",
            "pst": "ASD_pockets_sequence_binary_text_comp",
        },
    },
    {
        "panel_title": "Kinase",
        "source": "flat",
        "tasks": {
            "p":   "Kinase_pocket",
            "pt":  "Kinase_pocket_text",
            "ps":  "Kinase_combined",
            "pst": "Kinase_combined_text",
        },
    },
    {
        "panel_title": "PL8 + Kinase",
        "source": "flat",
        "tasks": {
            "p":   "merged_pocket_binary_comp",
            "pt":  "merged_pocket_binary_text_comp",
            "ps":  "merged_pocket_sequence_binary_comp",
            "pst": "merged_pocket_sequence_binary_text_comp",
        },
    },
    {
        "panel_title": "ASD + PL8 + Kinase",
        "source": "balanced",
        "tasks": {
            "p":   "ASD_merged_pocket_binary_comp",
            "pt":  "ASD_merged_pocket_binary_text_comp",
            "ps":  "ASD_merged_pocket_sequence_binary_comp",
            "pst": "ASD_merged_pocket_sequence_binary_text_comp",
        },
    },
]


# ── Loading helpers ───────────────────────────────────────────────────────────

def load_flat(base_dir: str, model: str, task: str, seed: int):
    path = os.path.join(base_dir, f"{task}_{model}_seed{seed}_preds.npz")
    return _load_npz(path)

def load_balanced_hierarchical(balanced_dir: str, model: str, task: str, seed: int):
    path = os.path.join(balanced_dir, model, f"{task}_balanced", f"seed{seed}", "preds.npz")
    return _load_npz(path)

def _load_npz(path: str):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr, sklearn_auc(fpr, tpr)

def average_roc(load_fn, root_dir: str, model: str, task: str):
    tprs, aucs = [], []
    for seed in SEEDS:
        result = load_fn(root_dir, model, task, seed)
        if result is None:
            continue
        fpr, tpr, roc_auc = result
        interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    if not tprs:
        return None

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    return {
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "n": len(tprs),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_figure(base_dir: str, balanced_dir: str, output_dir: str):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.5, 9.0),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for ax, dataset_cfg in zip(axes, DATASETS):
        panel_title = dataset_cfg["panel_title"]
        source = dataset_cfg["source"]
        tasks = dataset_cfg["tasks"]

        missing = []

        for emb_key in EMBEDDING_ORDER:
            task = tasks[emb_key]
            color = EMBEDDING_COLORS[emb_key]

            if source == "flat":
                result = average_roc(load_flat, base_dir, MODEL, task)
            elif source == "balanced":
                result = average_roc(load_balanced_hierarchical, balanced_dir, MODEL, task)
            else:
                raise ValueError(f"Unknown source type: {source}")

            if result is None:
                missing.append(task)
                continue

            mean_tpr = result["mean_tpr"]
            std_tpr = result["std_tpr"]
            mean_auc = result["mean_auc"]
            std_auc = result["std_auc"]
            n = result["n"]

            n_prefix = f"(n={n}) " if n < len(SEEDS) else ""
            label = f"{n_prefix}{EMBEDDING_LABELS[emb_key]}  {mean_auc:.2f}±{std_auc:.2f}"

            ax.plot(
                MEAN_FPR,
                mean_tpr,
                color=color,
                lw=2.0,
                label=label,
            )
            ax.fill_between(
                MEAN_FPR,
                np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color=color,
                alpha=0.12,
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate", fontsize=10)
        ax.set_ylabel("True positive rate", fontsize=10)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

        if missing:
            print(f"[{panel_title}] Missing tasks: {missing}")

    fig.suptitle(
        f"Representative ROC curves for {MODEL_LABEL} across datasets\n"
        f"(balanced ASD split; seeds {SEEDS[0]}–{SEEDS[-1]})",
        fontsize=14,
        fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "figure1_representative_md_no_graph_balancedASD_roc.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="<REPO_ROOT>/curve_plots",
        help="Directory with flat-format ROC .npz files.",
    )
    parser.add_argument(
        "--balanced_dir",
        default="<CURVE_PLOTS_ROOT>",
        help="Directory with hierarchical balanced ASD results.",
    )
    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/figure1_representative",
        help="Output directory for the figure.",
    )
    args = parser.parse_args()

    print(f"Base dir     : {args.base_dir}")
    print(f"Balanced dir : {args.balanced_dir}")
    print(f"Output dir   : {args.output_dir}")
    plot_figure(args.base_dir, args.balanced_dir, args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()