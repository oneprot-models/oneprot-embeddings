#!/usr/bin/env python3
"""
plot_auc_heatmaps_npz_with_excel_fill_publication_style.py

Publication-quality ROC-AUC heatmaps for four downstream embedding sets.

Primary source:
    preds.npz files from seeds 11-15

Fallback source:
    oneprot_allostery_results.xlsx

If a cell is missing from the npz files, the corresponding value is filled
from the Excel summary.

Outputs:
    auc_heatmaps_npz_with_excel_fill_per_embedding_publication.pdf
    auc_heatmaps_npz_with_excel_fill_per_embedding_publication.svg
    auc_heatmaps_npz_with_excel_fill_per_embedding_publication.png
    auc_heatmaps_npz_with_excel_fill_per_embedding_publication.tif

Usage:
    python plot_auc_heatmaps_npz_with_excel_fill_publication_style.py \
        --base_dir curve_plots \
        --balanced_dir curve_plots_balanced \
        --excel_file oneprot_allostery_results.xlsx \
        --output_dir separability_figures
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

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

MODEL_LABELS = {
    "oneprot_pocket_text_32900": "Pocket+Text",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "ST+SG+Pocket+Text",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "MD+ST+Pocket+Text",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "MD+SG+Pocket+Text",
    "oneprot_struct_graph_pocket_text_32900": "SG+Pocket+Text",
    "oneprot_md_combined_gpcr_32900": "MD+ST+SG+Pocket+Text",
    "oneprot_struct_token_pocket_text_32900": "ST+Pocket+Text",
}

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

EXCEL_DATASET_ORDER = [
    "PPI-site",
    "Kinsite",
    "DualSite",
    "AlloSite",
]

DATASET_TO_EXCEL = {
    "PL8": "PPI-site",
    "Kinase": "Kinsite",
    "PL8 + Kinase": "DualSite",
    "ASD + PL8 + Kinase": "AlloSite",
}

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket only",
    "pt": "Pocket + Text",
    "ps": "Pocket + Sequence",
    "pst": "Pocket + Sequence + Text",
}

EMBEDDING_ALIASES = {
    "Pockets": "p",
    "Pocket embeddings": "p",
    "Pocket": "p",

    "Pocket+text": "pt",
    "Pocket + Text": "pt",
    "Pocket and text embeddings": "pt",

    "Pocket+sequence": "ps",
    "Pocket + Sequence": "ps",
    "Pocket and sequence embeddings": "ps",

    "Pocket+sequence+text": "pst",
    "Pocket + Sequence + Text": "pst",
    "Pocket sequence and text embeddings": "pst",
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
PANEL_LABELS = ["A", "B", "C", "D"]


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def normalize_text(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def load_auc_from_npz(path):
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


def get_npz_path(base_dir, balanced_dir, dataset_name, embedding_key, model, seed):
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


def parse_excel_summary(excel_file):
    raw = pd.read_excel(excel_file, sheet_name=0, header=None)

    records = []
    dataset_names = set(EXCEL_DATASET_ORDER)
    model_names = set(MODELS)
    fallback_embedding_order = ["p", "ps", "pst", "pt"]

    current_dataset = None
    block_index = 0

    for i in range(len(raw)):
        first = normalize_text(raw.iloc[i, 0])

        if first in dataset_names:
            current_dataset = first
            block_index = 0
            continue

        if first != "Statistics by model_type:":
            continue

        if current_dataset is None:
            raise ValueError(f"Statistics block found before dataset near row {i + 1}")

        explicit_embedding = None

        for k in range(max(0, i - 6), i):
            candidate = normalize_text(raw.iloc[k, 0])
            if candidate in EMBEDDING_ALIASES:
                explicit_embedding = EMBEDDING_ALIASES[candidate]
                break

        if explicit_embedding is not None:
            embedding_key = explicit_embedding
        else:
            if block_index >= len(fallback_embedding_order):
                raise ValueError(
                    f"Could not infer embedding block for dataset {current_dataset}, "
                    f"block {block_index}, near row {i + 1}"
                )
            embedding_key = fallback_embedding_order[block_index]

        block_index += 1
        row = i + 1

        while row < len(raw):
            model_name = normalize_text(raw.iloc[row, 0])

            if model_name in dataset_names:
                break
            if model_name == "Statistics by model_type:":
                break

            if model_name in model_names:
                auc_value = raw.iloc[row, 1]
                std_value = raw.iloc[row, 2]
                n_value = raw.iloc[row, 3]

                records.append(
                    {
                        "dataset": current_dataset,
                        "embedding": embedding_key,
                        "model": model_name,
                        "auc": float(auc_value),
                        "std": float(std_value) if not pd.isna(std_value) else np.nan,
                        "n": int(n_value) if not pd.isna(n_value) else 0,
                    }
                )

            row += 1

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No model results were parsed from the Excel file.")

    print("\nParsed Excel counts by dataset and embedding:")
    print(df.groupby(["dataset", "embedding"]).size().unstack(fill_value=0))

    return df


def get_excel_value(excel_df, dataset_name, embedding_key, model):
    excel_dataset = DATASET_TO_EXCEL[dataset_name]

    hit = excel_df[
        (excel_df["dataset"] == excel_dataset)
        & (excel_df["embedding"] == embedding_key)
        & (excel_df["model"] == model)
    ]

    if hit.empty:
        return None, None

    auc_value = float(hit["auc"].iloc[0])
    n_value = int(hit["n"].iloc[0])

    return auc_value, n_value


def collect_matrix_for_embedding(base_dir, balanced_dir, excel_df, embedding_key):
    matrix = np.full(
        (len(DATASET_ORDER), len(MODELS)),
        np.nan,
        dtype=float,
    )

    n_matrix = np.zeros_like(matrix, dtype=int)
    source_matrix = np.full(matrix.shape, "missing", dtype=object)

    for i, dataset_name in enumerate(DATASET_ORDER):
        for j, model in enumerate(MODELS):
            aucs = []

            for seed in SEEDS:
                path = get_npz_path(
                    base_dir=base_dir,
                    balanced_dir=balanced_dir,
                    dataset_name=dataset_name,
                    embedding_key=embedding_key,
                    model=model,
                    seed=seed,
                )

                auc_value = load_auc_from_npz(path)

                if auc_value is not None:
                    aucs.append(auc_value)

            if aucs:
                matrix[i, j] = float(np.mean(aucs))
                n_matrix[i, j] = len(aucs)
                source_matrix[i, j] = "npz"
                continue

            excel_auc, excel_n = get_excel_value(
                excel_df=excel_df,
                dataset_name=dataset_name,
                embedding_key=embedding_key,
                model=model,
            )

            if excel_auc is not None:
                matrix[i, j] = excel_auc
                n_matrix[i, j] = excel_n if excel_n is not None else 0
                source_matrix[i, j] = "excel"

    return matrix, n_matrix, source_matrix


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


def style_heatmap_axis(ax, show_xlabels=True, show_ylabels=True):
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_yticks(np.arange(len(DATASET_ORDER)))

    if show_xlabels:
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODELS],
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=15,
            fontweight="bold",
        )
    else:
        ax.set_xticklabels([])

    if show_ylabels:
        ax.set_yticklabels(
            [DATASET_DISPLAY_NAMES[d] for d in DATASET_ORDER],
            fontsize=16,
            fontweight="bold",
        )
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="both", length=0, pad=4)
    bold_tick_labels(ax)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Thin white separators between cells for readability.
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)


def add_panel_label(ax, label):
    ax.text(
        -0.075,
        1.13,
        label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def annotate_cells(ax, matrix, source_matrix, vmin, vmax):
    for i in range(len(DATASET_ORDER)):
        for j in range(len(MODELS)):
            value = matrix[i, j]
            source = source_matrix[i, j]

            if np.isnan(value):
                text = "NA"
                text_color = "black"
            else:
                text = f"{value:.2f}"
                text_color = get_text_color(value, vmin, vmax)

            # Excel-filled values are shown with the same numeric format to avoid
            # visual clutter; source details are printed to stdout.
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=14.5,
                fontweight="bold",
                color=text_color,
            )


def plot_heatmaps(base_dir, balanced_dir, excel_file, output_dir):
    excel_df = parse_excel_summary(excel_file)

    matrices = {}
    n_matrices = {}
    source_matrices = {}

    for embedding_key in EMBEDDING_ORDER:
        matrix, n_matrix, source_matrix = collect_matrix_for_embedding(
            base_dir=base_dir,
            balanced_dir=balanced_dir,
            excel_df=excel_df,
            embedding_key=embedding_key,
        )

        matrices[embedding_key] = matrix
        n_matrices[embedding_key] = n_matrix
        source_matrices[embedding_key] = source_matrix

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(21.0, 12.0),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    axes = axes.flatten()

    vmin = 0.5
    vmax = 1.0
    cmap = "cividis"

    last_im = None

    for idx, (ax, embedding_key, panel_label) in enumerate(
        zip(axes, EMBEDDING_ORDER, PANEL_LABELS)
    ):
        matrix = matrices[embedding_key]
        source_matrix = source_matrices[embedding_key]

        last_im = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )

        add_panel_label(ax, panel_label)

        ax.set_title(
            EMBEDDING_LABELS[embedding_key],
            fontsize=20,
            fontweight="bold",
            color=TITLE_COLOR,
            pad=14,
        )

        row = idx // 2
        col = idx % 2
        style_heatmap_axis(
            ax,
            show_xlabels=row == 1,
            show_ylabels=True,
        )

        annotate_cells(
            ax=ax,
            matrix=matrix,
            source_matrix=source_matrix,
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle(
        "Mean ROC-AUC across separability regimes and model architectures",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.875,
        bottom=0.205,
        top=0.900,
        wspace=0.10,
        hspace=0.32,
    )

    cbar_ax = fig.add_axes([0.905, 0.255, 0.020, 0.56])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
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

    stem = os.path.join(
        output_dir,
        "auc_heatmaps_npz_with_excel_fill_per_embedding_publication",
    )

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

    print("\nExcel-filled cells:")

    for embedding_key in EMBEDDING_ORDER:
        source_matrix = source_matrices[embedding_key]
        matrix = matrices[embedding_key]

        for i, dataset_name in enumerate(DATASET_ORDER):
            for j, model in enumerate(MODELS):
                if source_matrix[i, j] == "excel":
                    print(
                        f"{embedding_key:>3} | "
                        f"{DATASET_DISPLAY_NAMES[dataset_name]:<12} | "
                        f"{MODEL_LABELS[model]:<25} | "
                        f"{matrix[i, j]:.3f}"
                    )


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
        "--excel_file",
        default="oneprot_allostery_results.xlsx",
        help="Excel summary file used only as fallback for missing cells.",
    )

    parser.add_argument(
        "--output_dir",
        default="separability_figures",
        help="Output directory.",
    )

    args = parser.parse_args()

    plot_heatmaps(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        excel_file=args.excel_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
