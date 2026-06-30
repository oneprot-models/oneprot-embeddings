"""
plot_auc_heatmaps_npz_with_excel_fill.py

Primary source:
    preds.npz files from seeds 11-15

Fallback source:
    oneprot_allostery_results.xlsx

If a cell is missing from the npz files, the corresponding value is filled
from the Excel summary.

Usage:
    python plot_auc_heatmaps_npz_with_excel_fill.py \
        --base_dir curve_plots \
        --balanced_dir curve_plots_balanced \
        --excel_file oneprot_allostery_results.xlsx \
        --output_dir separability_figures
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc as sklearn_auc


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
    "PL8 + Kinase": "DualSite",
    "ASD + PL8 + Kinase": "AlloSite",
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
    "p": "Pocket",
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


def normalize_text(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def load_auc_from_npz(path):
    if not os.path.exists(path):
        return None

    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

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


def get_text_color(value, vmin, vmax):
    if np.isnan(value):
        return "black"

    norm_value = (value - vmin) / (vmax - vmin)

    if norm_value < 0.45:
        return "white"

    return "black"


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
        figsize=(17.2, 9.8),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()
    panel_letters = ["A", "B", "C", "D"]

    vmin = 0.5
    vmax = 1.0
    cmap = "cividis"

    last_im = None

    for ax, embedding_key, panel_letter in zip(
        axes,
        EMBEDDING_ORDER,
        panel_letters,
    ):
        matrix = matrices[embedding_key]
        source_matrix = source_matrices[embedding_key]

        last_im = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

        ax.set_title(
            f"{panel_letter}. {EMBEDDING_LABELS[embedding_key]}",
            fontsize=16,
            fontweight="bold",
            pad=8,
        )

        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODELS],
            rotation=35,
            ha="right",
            fontsize=12,
        )

        ax.set_yticks(np.arange(len(DATASET_ORDER)))
        ax.set_yticklabels(
            [DATASET_DISPLAY_NAMES[d] for d in DATASET_ORDER],
            fontsize=14,
            fontweight="bold",
        )

        for i in range(len(DATASET_ORDER)):
            for j in range(len(MODELS)):
                value = matrix[i, j]
                source = source_matrix[i, j]

                if np.isnan(value):
                    text = "NA"
                    text_color = "black"
                else:
                    text = f"{value:.2f}"
                    if source == "excel":
                        text = f"{text}"

                    text_color = get_text_color(value, vmin, vmax)

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_color,
                )

    fig.suptitle(
        "Mean ROC-AUC across separability regimes and model architectures",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.18,
        top=0.90,
        wspace=0.08,
        hspace=0.24,
    )

    cbar_ax = fig.add_axes([0.905, 0.24, 0.018, 0.58])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Mean ROC-AUC", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # fig.text(
    #     0.08,
    #     0.055,
    #     "* Values marked with an asterisk were filled from the 10-seed Excel summary because the corresponding seed 11-15 prediction files were unavailable.",
    #     ha="left",
    #     va="center",
    #     fontsize=10,
    # )

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(
        output_dir,
        "auc_heatmaps_npz_with_excel_fill_per_embedding.png",
    )

    out_pdf = os.path.join(
        output_dir,
        "auc_heatmaps_npz_with_excel_fill_per_embedding.pdf",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG -> {out_png}")
    print(f"Saved PDF -> {out_pdf}")

    print("\nExcel-filled cells:")

    for embedding_key in EMBEDDING_ORDER:
        source_matrix = source_matrices[embedding_key]
        matrix = matrices[embedding_key]

        for i, dataset_name in enumerate(DATASET_ORDER):
            for j, model in enumerate(MODELS):
                if source_matrix[i, j] == "excel":
                    print(
                        f"{embedding_key:>3} | "
                        f"{DATASET_DISPLAY_NAMES[dataset_name]:<10} | "
                        f"{MODEL_LABELS[model]:<25} | "
                        f"{matrix[i, j]:.3f}"
                    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        default="curve_plots",
    )

    parser.add_argument(
        "--balanced_dir",
        default="curve_plots_balanced",
    )

    parser.add_argument(
        "--excel_file",
        default="oneprot_allostery_results.xlsx",
    )

    parser.add_argument(
        "--output_dir",
        default="separability_figures",
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