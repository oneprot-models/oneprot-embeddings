#!/usr/bin/env python3
"""
plot_auc_heatmap_dataset_by_embedding_publication_style.py

Publication-quality single ROC-AUC heatmap:
    rows    = datasets / separability regimes
    columns = downstream embedding combinations

Each cell reports the mean ROC-AUC averaged over all listed model architectures
and seeds 11-15.

Styled to match the publication-style heatmap figures:
  - DejaVu Sans font
  - bold, large labels
  - cividis color scale
  - white cell separators
  - bold cell annotations
  - high-resolution PDF, SVG, PNG, and TIFF outputs

Usage:
    python plot_auc_heatmap_dataset_by_embedding_publication_style.py \
        --base_dir curve_plots \
        --balanced_dir curve_plots_balanced \
        --output_dir separability_figures
"""

import os
import argparse
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc as sklearn_auc


# -----------------------------------------------------------------------------
# Publication-style matplotlib settings
# -----------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "axes.linewidth": 1.8,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 20,
    "axes.titleweight": "bold",

    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,

    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODELS = [
    "oneprot_pocket_text_32900",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity",
    "oneprot_md_combined_gpcr_no_struct_graph_32900",
    "oneprot_md_combined_gpcr_no_struct_token_32900",
    "oneprot_struct_graph_pocket_text_32900",
    "oneprot_md_combined_gpcr_32900",
    "oneprot_struct_token_pocket_text_32900",
]

SEEDS = [11, 12, 13, 14, 15]

DATASET_ORDER = [
    "PL8",
    "Kinase",
    "PL8 + Kinase",
    "ASD + PL8 + Kinase",
]

DATASET_DISPLAY_NAMES = {
    "PL8": "PPI-Site",
    "Kinase": "KinSite",
    "PL8 + Kinase": "Dual-Site",
    "ASD + PL8 + Kinase": "AlloDiverse",
}

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket only",
    "pt": "Pocket + Text",
    "ps": "Pocket + Sequence",
    "pst": "Pocket + Sequence + Text",
}

TASK_MAP = {
    "p": {
        "PL8": "ASD_pockets_binary_comp",
        "Kinase": "Kinase_pocket",
        "PL8 + Kinase": "merged_pocket_binary_comp",
        "ASD + PL8 + Kinase": "ASD_merged_pocket_binary_comp",
    },
    "pt": {
        "PL8": "ASD_pockets_binary_text_comp",
        "Kinase": "Kinase_pocket_text",
        "PL8 + Kinase": "merged_pocket_binary_text_comp",
        "ASD + PL8 + Kinase": "ASD_merged_pocket_binary_text_comp",
    },
    "ps": {
        "PL8": "ASD_pockets_sequence_binary_comp",
        "Kinase": "Kinase_combined",
        "PL8 + Kinase": "merged_pocket_sequence_binary_comp",
        "ASD + PL8 + Kinase": "ASD_merged_pocket_sequence_binary_comp",
    },
    "pst": {
        "PL8": "ASD_pockets_sequence_binary_text_comp",
        "Kinase": "Kinase_combined_text",
        "PL8 + Kinase": "merged_pocket_sequence_binary_text_comp",
        "ASD + PL8 + Kinase": "ASD_merged_pocket_sequence_binary_text_comp",
    },
}

TITLE_COLOR = "#111111"


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def load_auc(path: str):
    if not os.path.exists(path):
        return None

    try:
        data = np.load(path)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(sklearn_auc(fpr, tpr))


def get_path(base_dir: str, balanced_dir: str, dataset_name: str, embedding_key: str, model: str, seed: int):
    task = TASK_MAP[embedding_key][dataset_name]

    if dataset_name == "ASD + PL8 + Kinase":
        return os.path.join(
            balanced_dir,
            model,
            f"{task}_balanced",
            f"seed{seed}",
            "preds.npz",
        )

    return os.path.join(
        base_dir,
        f"{task}_{model}_seed{seed}_preds.npz",
    )


def collect_mean_auc_matrix(base_dir: str, balanced_dir: str):
    matrix = np.full(
        (len(DATASET_ORDER), len(EMBEDDING_ORDER)),
        np.nan,
        dtype=float,
    )

    counts = np.zeros_like(matrix, dtype=int)

    for i, dataset_name in enumerate(DATASET_ORDER):
        for j, embedding_key in enumerate(EMBEDDING_ORDER):
            aucs = []

            for model in MODELS:
                for seed in SEEDS:
                    path = get_path(
                        base_dir=base_dir,
                        balanced_dir=balanced_dir,
                        dataset_name=dataset_name,
                        embedding_key=embedding_key,
                        model=model,
                        seed=seed,
                    )

                    auc_value = load_auc(path)

                    if auc_value is not None:
                        aucs.append(auc_value)

            if aucs:
                matrix[i, j] = float(np.mean(aucs))
                counts[i, j] = len(aucs)

    return matrix, counts


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def get_text_color(value, vmin, vmax):
    if np.isnan(value):
        return "black"

    norm_value = (value - vmin) / (vmax - vmin)
    return "white" if norm_value < 0.43 else "black"


def bold_tick_labels(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def style_heatmap_axis(ax):
    ax.set_xticks(np.arange(len(EMBEDDING_ORDER)))
    ax.set_xticklabels(
        [EMBEDDING_LABELS[k] for k in EMBEDDING_ORDER],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels(
        [DATASET_DISPLAY_NAMES[d] for d in DATASET_ORDER],
        fontsize=17,
        fontweight="bold",
    )

    ax.tick_params(axis="both", length=0, pad=7)
    bold_tick_labels(ax)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Thin white separators between cells for readability.
    ax.set_xticks(np.arange(-0.5, len(EMBEDDING_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)


def annotate_cells(ax, matrix, counts, vmin, vmax, show_counts=False):
    for i in range(len(DATASET_ORDER)):
        for j in range(len(EMBEDDING_ORDER)):
            value = matrix[i, j]
            n = counts[i, j]

            if np.isnan(value):
                text = "NA"
                text_color = "black"
            else:
                if show_counts:
                    text = f"{value:.2f}\n(n={n})"
                else:
                    text = f"{value:.2f}"
                text_color = get_text_color(value, vmin, vmax)

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=17 if not show_counts else 14.5,
                fontweight="bold",
                color=text_color,
                linespacing=1.05,
            )


def plot_heatmap(base_dir: str, balanced_dir: str, output_dir: str, show_counts: bool = False):
    matrix, counts = collect_mean_auc_matrix(base_dir, balanced_dir)

    vmin = 0.5
    vmax = 1.0
    cmap = "cividis"

    fig, ax = plt.subplots(figsize=(11.2, 8.2), constrained_layout=False)

    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    style_heatmap_axis(ax)
    annotate_cells(ax, matrix, counts, vmin, vmax, show_counts=show_counts)

    ax.set_title(
        "Mean ROC-AUC across datasets and embedding combinations",
        fontsize=21,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=18,
    )

    fig.subplots_adjust(
        left=0.205,
        right=0.850,
        bottom=0.235,
        top=0.875,
    )

    cbar_ax = fig.add_axes([0.885, 0.255, 0.026, 0.56])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(
        "Mean ROC-AUC",
        fontsize=18,
        fontweight="bold",
        labelpad=12,
    )
    cbar.ax.tick_params(labelsize=15, width=1.5, length=5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    cbar.outline.set_linewidth(1.5)

    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.join(output_dir, "auc_heatmap_dataset_by_embedding_publication")

    out_pdf = f"{stem}.pdf"
    out_svg = f"{stem}.svg"
    out_png = f"{stem}.png"
    out_tif = f"{stem}.tif"

    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_tif, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"Saved PDF  -> {out_pdf}")
    print(f"Saved SVG  -> {out_svg}")
    print(f"Saved PNG  -> {out_png}")
    print(f"Saved TIFF -> {out_tif}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        default="curve_plots",
        help="Directory with flat-format prediction .npz files.",
    )

    parser.add_argument(
        "--balanced_dir",
        default="curve_plots_balanced",
        help="Directory with hierarchical balanced AlloDiverse prediction files.",
    )

    parser.add_argument(
        "--output_dir",
        default="separability_figures",
        help="Output directory.",
    )

    parser.add_argument(
        "--show_counts",
        action="store_true",
        help="Display the number of contributing model/seed runs inside each cell.",
    )

    args = parser.parse_args()

    plot_heatmap(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
        show_counts=args.show_counts,
    )


if __name__ == "__main__":
    main()
