"""
plot_auc_heatmap_dataset_by_embedding.py
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

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket",
    "pt": "Pocket+Text",
    "ps": "Pocket+Sequence",
    "pst": "Pocket+Sequence+Text",
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
                        base_dir,
                        balanced_dir,
                        dataset_name,
                        embedding_key,
                        model,
                        seed,
                    )
                    auc_value = load_auc(path)

                    if auc_value is not None:
                        aucs.append(auc_value)

            if aucs:
                matrix[i, j] = np.mean(aucs)
                counts[i, j] = len(aucs)

    return matrix, counts


def plot_heatmap(base_dir, balanced_dir, output_dir):
    matrix, counts = collect_mean_auc_matrix(base_dir, balanced_dir)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    im = ax.imshow(
        matrix,
        vmin=0.5,
        vmax=1.0,
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean ROC-AUC", fontsize=11)

    ax.set_xticks(np.arange(len(EMBEDDING_ORDER)))
    ax.set_xticklabels(
        [EMBEDDING_LABELS[k] for k in EMBEDDING_ORDER],
        rotation=25,
        ha="right",
    )

    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels(DATASET_ORDER)

    for i in range(len(DATASET_ORDER)):
        for j in range(len(EMBEDDING_ORDER)):
            value = matrix[i, j]
            n = counts[i, j]

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
                fontsize=9,
            )

    ax.set_title(
        "Mean ROC-AUC across datasets and embedding combinations",
        fontsize=13,
        fontweight="bold",
    )

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
        default="<REPO_ROOT>/curve_plots",
    )
    parser.add_argument(
        "--balanced_dir",
        default="<CURVE_PLOTS_ROOT>",
    )
    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/separability_figures",
    )

    args = parser.parse_args()

    plot_heatmap(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()