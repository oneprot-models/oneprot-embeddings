"""
plot_auc_heatmap_dataset_by_embedding.py

Mean ROC-AUC heatmap:
    rows    = datasets
    columns = embedding combinations

Styled to match plot_auc_heatmaps_npz_with_excel_fill.py
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

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket",
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


def collect_mean_auc_matrix(base_dir, balanced_dir):
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


def get_text_color(value, vmin, vmax):
    if np.isnan(value):
        return "black"

    norm_value = (value - vmin) / (vmax - vmin)

    if norm_value < 0.45:
        return "white"

    return "black"


def plot_heatmap(base_dir, balanced_dir, output_dir):
    matrix, counts = collect_mean_auc_matrix(base_dir, balanced_dir)

    vmin = 0.5
    vmax = 1.0
    cmap = "cividis"

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(EMBEDDING_ORDER)))
    ax.set_xticklabels(
        [EMBEDDING_LABELS[k] for k in EMBEDDING_ORDER],
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
        for j in range(len(EMBEDDING_ORDER)):
            value = matrix[i, j]

            if np.isnan(value):
                text = "NA"
                text_color = "black"
            else:
                text = f"{value:.2f}"
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

    ax.set_title(
        "Mean ROC-AUC across datasets and embedding combinations",
        fontsize=20,
        fontweight="bold",
        pad=14,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean ROC-AUC", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(output_dir, "auc_heatmap_dataset_by_embedding.png")
    out_pdf = os.path.join(output_dir, "auc_heatmap_dataset_by_embedding.pdf")

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

    plot_heatmap(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()