"""
plot_figure3_tpr_tnr.py

Figure 3: threshold-dependent behaviour for a representative model.

Design:
    - fixed model: md_no_graph
    - 2x2 panels:
        (1) ASD+PL8+Kinase  : TPR, balanced vs unbalanced
        (2) ASD+PL8+Kinase  : TNR, balanced vs unbalanced
        (3) Kinase          : TPR
        (4) Kinase          : TNR
    - x-axis = embedding sets:
        pocket
        pocket+text
        pocket+sequence
        pocket+sequence+text
    - balanced ASD results loaded from hierarchical balanced folder
    - unbalanced ASD + Kinase loaded from flat files

Files:
    Unbalanced flat:
        <unbalanced_dir>/{task}_{model}_seed{seed}_preds.npz

    Balanced hierarchical:
        <balanced_dir>/<model>/{task}_balanced/seed{seed}/preds.npz

Usage:
    python plot_figure3_tpr_tnr.py \
        --unbalanced_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots \
        --balanced_dir /p/scratch/hai_oneprot/curve_plots \
        --output_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/figure3_tpr_tnr \
        --threshold 0.5
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "oneprot_md_combined_gpcr_no_struct_graph_32900"
MODEL_LABEL = "md_no_graph"

SEEDS = [11, 12, 13, 14, 15]

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]
EMBEDDING_LABELS = {
    "p": "pocket",
    "pt": "pocket+text",
    "ps": "pocket+sequence",
    "pst": "pocket+sequence+text",
}

# ASD merged tasks
ASD_TASKS = {
    "p":   "ASD_merged_pocket_binary_comp",
    "pt":  "ASD_merged_pocket_binary_text_comp",
    "ps":  "ASD_merged_pocket_sequence_binary_comp",
    "pst": "ASD_merged_pocket_sequence_binary_text_comp",
}

# Kinase tasks
KINASE_TASKS = {
    "p":   "Kinase_pocket",
    "pt":  "Kinase_pocket_text",
    "ps":  "Kinase_combined",
    "pst": "Kinase_combined_text",
}

# Colors for embedding groups
EMBEDDING_COLORS = {
    "p":   "#1f77b4",
    "pt":  "#ff7f0e",
    "ps":  "#2ca02c",
    "pst": "#d62728",
}

# Bar colors for regimes
COLOR_UNBAL = "#cc4c4c"
COLOR_BAL = "#4c72b0"
COLOR_KIN = "#7a5aa6"


# ── Loading ───────────────────────────────────────────────────────────────────

def load_flat(unbalanced_dir: str, model: str, task: str, seed: int):
    path = os.path.join(unbalanced_dir, f"{task}_{model}_seed{seed}_preds.npz")
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
    return y_true, y_pred


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_tpr_tnr(y_true, y_pred, threshold=0.5):
    y_hat = (y_pred >= threshold).astype(int)

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    tp = np.sum((y_hat == 1) & pos_mask)
    fn = np.sum((y_hat == 0) & pos_mask)
    tn = np.sum((y_hat == 0) & neg_mask)
    fp = np.sum((y_hat == 1) & neg_mask)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return tpr, tnr

def average_metric(load_fn, root_dir: str, model: str, task: str, threshold=0.5):
    tprs, tnrs = [], []

    for seed in SEEDS:
        loaded = load_fn(root_dir, model, task, seed)
        if loaded is None:
            continue
        y_true, y_pred = loaded
        tpr, tnr = compute_tpr_tnr(y_true, y_pred, threshold=threshold)
        if not np.isnan(tpr):
            tprs.append(tpr)
        if not np.isnan(tnr):
            tnrs.append(tnr)

    if len(tprs) == 0 or len(tnrs) == 0:
        return None

    return {
        "mean_tpr": float(np.mean(tprs)),
        "std_tpr": float(np.std(tprs)),
        "mean_tnr": float(np.mean(tnrs)),
        "std_tnr": float(np.std(tnrs)),
        "n": min(len(tprs), len(tnrs)),
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def add_bar_labels(ax, bars, values):
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

def style_axis(ax, ylabel, title):
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=20, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_figure(unbalanced_dir: str, balanced_dir: str, output_dir: str, threshold: float):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), constrained_layout=True)
    ax_asd_tpr, ax_asd_tnr, ax_kin_tpr, ax_kin_tnr = axes.flatten()

    x = np.arange(len(EMBEDDING_ORDER))
    width = 0.36

    # ── ASD merged: balanced vs unbalanced ───────────────────────────────────
    asd_tpr_unbal, asd_tpr_bal = [], []
    asd_tnr_unbal, asd_tnr_bal = [], []

    for emb in EMBEDDING_ORDER:
        task = ASD_TASKS[emb]

        res_unbal = average_metric(load_flat, unbalanced_dir, MODEL, task, threshold=threshold)
        res_bal = average_metric(load_balanced_hierarchical, balanced_dir, MODEL, task, threshold=threshold)

        if res_unbal is None or res_bal is None:
            print(f"[ASD][{emb}] Missing data")
            asd_tpr_unbal.append(np.nan)
            asd_tpr_bal.append(np.nan)
            asd_tnr_unbal.append(np.nan)
            asd_tnr_bal.append(np.nan)
            continue

        asd_tpr_unbal.append(res_unbal["mean_tpr"])
        asd_tpr_bal.append(res_bal["mean_tpr"])
        asd_tnr_unbal.append(res_unbal["mean_tnr"])
        asd_tnr_bal.append(res_bal["mean_tnr"])

    bars1 = ax_asd_tpr.bar(x - width/2, asd_tpr_unbal, width, color=COLOR_UNBAL, label="unbalanced")
    bars2 = ax_asd_tpr.bar(x + width/2, asd_tpr_bal, width, color=COLOR_BAL, label="balanced")
    ax_asd_tpr.set_xticks(x)
    ax_asd_tpr.set_xticklabels([EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER])
    style_axis(ax_asd_tpr, "True positive rate", "ASD + PL8 + Kinase")
    ax_asd_tpr.legend(fontsize=9, framealpha=0.9)
    add_bar_labels(ax_asd_tpr, bars1, asd_tpr_unbal)
    add_bar_labels(ax_asd_tpr, bars2, asd_tpr_bal)

    bars3 = ax_asd_tnr.bar(x - width/2, asd_tnr_unbal, width, color=COLOR_UNBAL, label="unbalanced")
    bars4 = ax_asd_tnr.bar(x + width/2, asd_tnr_bal, width, color=COLOR_BAL, label="balanced")
    ax_asd_tnr.set_xticks(x)
    ax_asd_tnr.set_xticklabels([EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER])
    style_axis(ax_asd_tnr, "True negative rate", "ASD + PL8 + Kinase")
    ax_asd_tnr.legend(fontsize=9, framealpha=0.9)
    add_bar_labels(ax_asd_tnr, bars3, asd_tnr_unbal)
    add_bar_labels(ax_asd_tnr, bars4, asd_tnr_bal)

    # ── Kinase: unbalanced only ───────────────────────────────────────────────
    kin_tpr, kin_tnr = [], []

    for emb in EMBEDDING_ORDER:
        task = KINASE_TASKS[emb]
        res = average_metric(load_flat, unbalanced_dir, MODEL, task, threshold=threshold)

        if res is None:
            print(f"[Kinase][{emb}] Missing data")
            kin_tpr.append(np.nan)
            kin_tnr.append(np.nan)
            continue

        kin_tpr.append(res["mean_tpr"])
        kin_tnr.append(res["mean_tnr"])

    bars5 = ax_kin_tpr.bar(x, kin_tpr, width=0.55, color=COLOR_KIN)
    ax_kin_tpr.set_xticks(x)
    ax_kin_tpr.set_xticklabels([EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER])
    style_axis(ax_kin_tpr, "True positive rate", "Kinase")
    add_bar_labels(ax_kin_tpr, bars5, kin_tpr)

    bars6 = ax_kin_tnr.bar(x, kin_tnr, width=0.55, color=COLOR_KIN)
    ax_kin_tnr.set_xticks(x)
    ax_kin_tnr.set_xticklabels([EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER])
    style_axis(ax_kin_tnr, "True negative rate", "Kinase")
    add_bar_labels(ax_kin_tnr, bars6, kin_tnr)

    fig.suptitle(
        f"Threshold-dependent behaviour for {MODEL_LABEL} (threshold={threshold})\n"
        f"Balanced training increases sensitivity in ASD + PL8 + Kinase, while Kinase remains specificity-dominated",
        fontsize=13,
        fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "figure3_tpr_tnr_md_no_graph.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unbalanced_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
    )
    parser.add_argument(
        "--balanced_dir",
        default="/p/scratch/hai_oneprot/curve_plots",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/figure3_tpr_tnr",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for converting probabilities to class labels.",
    )
    args = parser.parse_args()

    print(f"Unbalanced dir : {args.unbalanced_dir}")
    print(f"Balanced dir   : {args.balanced_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Model          : {MODEL_LABEL}")
    print(f"Threshold      : {args.threshold}")

    plot_figure(
        unbalanced_dir=args.unbalanced_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
    print("Done.")

if __name__ == "__main__":
    main()