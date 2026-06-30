"""
plot_auc_heatmaps_from_excel.py

Creates a 2x2 panel of heatmaps from oneprot_allostery_results.xlsx.

Each panel corresponds to one embedding combination:
    p   = Pocket
    pt  = Pocket + Text
    ps  = Pocket + Sequence
    pst = Pocket + Sequence + Text

Rows:
    datasets from the Excel file

Columns:
    model architectures

Each cell:
    Average_auc from the Excel file

Usage:
    python plot_auc_heatmaps_from_excel.py \
        --excel_file /mnt/data/oneprot_allostery_results.xlsx \
        --output_dir separability_figures
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "Full all-atom",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "MD no graph",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "MD no token",
    "oneprot_struct_graph_pocket_text_32900": "Struct graph",
    "oneprot_md_combined_gpcr_32900": "MD combined",
    "oneprot_struct_token_pocket_text_32900": "Struct token",
}

DATASET_ORDER = [
    "PPI-site",
    "Kinsite",
    "DualSite",
    "AlloSite",
]

DATASET_LABELS = {
    "PPI-site": "PL8",
    "Kinsite": "Kinase",
    "DualSite": "PL8 + Kinase",
    "AlloSite": "ASD + PL8 + Kinase",
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

    "Pocket+text": "pt",
    "Pocket and text embeddings": "pt",

    "Pocket+sequence": "ps",
    "Pocket and sequence embeddings": "ps",

    "Pocket+sequence+text": "pst",
    "Pocket sequence and text embeddings": "pst",
}


def normalize_text(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def parse_excel_summary(excel_file):
    raw = pd.read_excel(excel_file, sheet_name=0, header=None)

    records = []

    dataset_names = set(DATASET_ORDER)
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

        # Try to find an explicit embedding label just above this block
        explicit_embedding = None
        for k in range(max(0, i - 6), i):
            candidate = normalize_text(raw.iloc[k, 0])
            if candidate in EMBEDDING_ALIASES:
                explicit_embedding = EMBEDDING_ALIASES[candidate]
                break

        # If no explicit label exists, use block order
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

        # Find model rows after the Statistics block
        row = i + 1
        while row < len(raw):
            model_name = normalize_text(raw.iloc[row, 0])

            # Stop when next dataset or next statistics block begins
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

    print("\nParsed counts by dataset and embedding:")
    print(
        df.groupby(["dataset", "embedding"])
        .size()
        .unstack(fill_value=0)
    )

    return df


def make_matrix(df, embedding_key):
    matrix = np.full(
        (len(DATASET_ORDER), len(MODELS)),
        np.nan,
        dtype=float,
    )

    n_matrix = np.zeros_like(matrix, dtype=int)

    sub = df[df["embedding"] == embedding_key]

    for i, dataset in enumerate(DATASET_ORDER):
        for j, model in enumerate(MODELS):
            hit = sub[
                (sub["dataset"] == dataset) &
                (sub["model"] == model)
            ]

            if not hit.empty:
                matrix[i, j] = hit["auc"].iloc[0]
                n_matrix[i, j] = int(hit["n"].iloc[0])

    return matrix, n_matrix


def plot_heatmaps(df, output_dir):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 8.5),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()
    panel_letters = ["A", "B", "C", "D"]

    vmin = 0.5
    vmax = 1.0
    last_im = None

    for ax, embedding_key, panel_letter in zip(
        axes,
        EMBEDDING_ORDER,
        panel_letters,
    ):
        matrix, n_matrix = make_matrix(df, embedding_key)

        last_im = ax.imshow(
            matrix,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

        ax.set_title(
            f"{panel_letter}. {EMBEDDING_LABELS[embedding_key]}",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODELS],
            rotation=35,
            ha="right",
            fontsize=9,
        )

        ax.set_yticks(np.arange(len(DATASET_ORDER)))
        ax.set_yticklabels(
            [DATASET_LABELS[d] for d in DATASET_ORDER],
            fontsize=10,
        )

        for i in range(len(DATASET_ORDER)):
            for j in range(len(MODELS)):
                value = matrix[i, j]
                n = n_matrix[i, j]

                if np.isnan(value):
                    text = "NA"
                else:
                    text = f"{value:.2f}\n(n={n})"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )

    fig.suptitle(
        "Mean ROC-AUC across datasets and model architectures for each embedding combination",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Mean ROC-AUC", fontsize=11)

    fig.tight_layout(rect=[0, 0, 0.96, 0.95])

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(
        output_dir,
        "auc_heatmaps_from_excel_per_embedding.png",
    )
    out_pdf = os.path.join(
        output_dir,
        "auc_heatmaps_from_excel_per_embedding.pdf",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG -> {out_png}")
    print(f"Saved PDF -> {out_pdf}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--excel_file",
        #required=True,
        default="oneprot_allostery_results.xlsx",
    )

    parser.add_argument(
        "--output_dir",
        default="separability_figures",
    )

    args = parser.parse_args()

    df = parse_excel_summary(args.excel_file)

    print("Parsed rows:")
    print(df.head())
    print(f"Total parsed records: {len(df)}")

    plot_heatmaps(df, args.output_dir)


if __name__ == "__main__":
    main()