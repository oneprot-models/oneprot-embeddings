"""
plot_auc_heatmaps_by_embedding_and_model.py

Creates a 2x2 panel of heatmaps.

Each panel corresponds to one extracted embedding combination:
    p   = Pocket
    pt  = Pocket + Text
    ps  = Pocket + Sequence
    pst = Pocket + Sequence + Text

Rows:
    datasets

Columns:
    model architectures

Each cell:
    mean ROC-AUC across seeds

Usage:
    python plot_auc_heatmaps_by_embedding_and_model.py \
        --base_dir curve_plots \
        --balanced_dir curve_plots_balanced \
        --output_dir separability_figures
"""

import os
import argparse
import numpy as np

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
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "Full all-atom",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "MD no graph",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "MD no token",
    "oneprot_struct_graph_pocket_text_32900": "Struct graph",
    "oneprot_md_combined_gpcr_32900": "MD combined",
    "oneprot_struct_token_pocket_text_32900": "Struct token",
}

SEEDS = [11, 12, 13, 14, 15]

DATASET_ORDER = [
    "PPI-Site",
    "KinSite",
    "DualSite",
    "AlloSite",
]

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket",
    "pt": "Pocket + Text",
    "ps": "Pocket + Sequence",
    "pst": "Pocket + Sequence + Text",
}

TASK_MAP = {
    "p": {
        "PPI-Site": "ASD_pockets_binary_comp",
        "KinSite": "Kinase_pocket",
        "DualSite": "merged_pocket_binary_comp",
        "AlloSite": "ASD_merged_pocket_binary_comp",
    },
    "pt": {
        "PPI-Site": "ASD_pockets_binary_text_comp",
        "KinSite": "Kinase_pocket_text",
        "DualSite": "merged_pocket_binary_text_comp",
        "AlloSite": "ASD_merged_pocket_binary_text_comp",
    },
    "ps": {
        "PPI-Site": "ASD_pockets_sequence_binary_comp",
        "KinSite": "Kinase_combined",
        "DualSite": "merged_pocket_sequence_binary_comp",
        "AlloSite": "ASD_merged_pocket_sequence_binary_comp",
    },
    "pst": {
        "PPI-Site": "ASD_pockets_sequence_binary_text_comp",
        "KinSite": "Kinase_combined_text",
        "DualSite": "merged_pocket_sequence_binary_text_comp",
        "AlloSite": "ASD_merged_pocket_sequence_binary_text_comp",
    },
}


def load_auc(path):
    if not os.path.exists(path):
        return None

    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(sklearn_auc(fpr, tpr))


def get_path(base_dir, balanced_dir, dataset_name, embedding_key, model, seed):
    task = TASK_MAP[embedding_key][dataset_name]

    if dataset_name == "AlloSite":
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


def collect_matrix_for_embedding(base_dir, balanced_dir, embedding_key):
    matrix = np.full(
        (len(DATASET_ORDER), len(MODELS)),
        np.nan,
        dtype=float,
    )

    counts = np.zeros_like(matrix, dtype=int)

    for i, dataset_name in enumerate(DATASET_ORDER):
        for j, model in enumerate(MODELS):
            aucs = []

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
                matrix[i, j] = np.mean(aucs)
                counts[i, j] = len(aucs)

    return matrix, counts


def plot_heatmaps(base_dir, balanced_dir, output_dir):
    matrices = {}
    counts = {}

    for embedding_key in EMBEDDING_ORDER:
        matrix, count_matrix = collect_matrix_for_embedding(
            base_dir=base_dir,
            balanced_dir=balanced_dir,
            embedding_key=embedding_key,
        )
        matrices[embedding_key] = matrix
        counts[embedding_key] = count_matrix

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
        matrix = matrices[embedding_key]
        count_matrix = counts[embedding_key]

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
        ax.set_yticklabels(DATASET_ORDER, fontsize=10)

        for i in range(len(DATASET_ORDER)):
            for j in range(len(MODELS)):
                value = matrix[i, j]
                n = count_matrix[i, j]

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
        "auc_heatmaps_dataset_by_model_per_embedding.png",
    )
    out_pdf = os.path.join(
        output_dir,
        "auc_heatmaps_dataset_by_model_per_embedding.pdf",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved PNG -> {out_png}")
    print(f"Saved PDF -> {out_pdf}")


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
        "--output_dir",
        default="separability_figures",
    )

    args = parser.parse_args()

    plot_heatmaps(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()