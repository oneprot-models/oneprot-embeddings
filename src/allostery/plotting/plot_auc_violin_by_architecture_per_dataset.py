"""
plot_auc_violin_by_architecture_per_dataset.py

Diagnostic figure:
    - 4 panels = datasets
    - x-axis = encoder architecture / OneProt model variant
    - y-axis = ROC-AUC
    - each violin pools all embedding combinations and seeds
    - overlaid points are colored by embedding combination

Usage:
    python plot_auc_violin_by_architecture_per_dataset.py \
        --base_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots \
        --balanced_dir /p/scratch/hai_oneprot/curve_plots \
        --output_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/separability_figures
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

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket",
    "pt": "Pocket+Text",
    "ps": "Pocket+Sequence",
    "pst": "Pocket+Sequence+Text",
}

EMBEDDING_COLORS = {
    "p": "#1f77b4",
    "pt": "#ff7f0e",
    "ps": "#2ca02c",
    "pst": "#d62728",
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


def collect_auc_values(base_dir, balanced_dir):
    rows = []

    for dataset_name in DATASET_ORDER:
        for model in MODELS:
            for embedding_key in EMBEDDING_ORDER:
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

                    if auc_value is None:
                        continue

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "embedding": embedding_key,
                            "embedding_label": EMBEDDING_LABELS[embedding_key],
                            "seed": seed,
                            "auc": auc_value,
                        }
                    )

    return rows


def plot_architecture_violins(base_dir, balanced_dir, output_dir):
    rows = collect_auc_values(base_dir, balanced_dir)

    if not rows:
        raise RuntimeError("No AUC values loaded. Check input directories.")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17.5, 10.5),
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.flatten()
    positions = np.arange(1, len(MODELS) + 1)

    rng = np.random.default_rng(123)

    for ax, dataset_name in zip(axes, DATASET_ORDER):
        data_by_model = []

        for model in MODELS:
            aucs = [
                r["auc"]
                for r in rows
                if r["dataset"] == dataset_name and r["model"] == model
            ]
            data_by_model.append(np.array(aucs, dtype=float))

        ax.violinplot(
            data_by_model,
            positions=positions,
            showmeans=True,
            showmedians=True,
            showextrema=True,
            widths=0.75,
        )

        for i, model in enumerate(MODELS, start=1):
            model_rows = [
                r
                for r in rows
                if r["dataset"] == dataset_name and r["model"] == model
            ]

            for embedding_key in EMBEDDING_ORDER:
                emb_rows = [
                    r for r in model_rows if r["embedding"] == embedding_key
                ]

                if not emb_rows:
                    continue

                aucs = np.array([r["auc"] for r in emb_rows], dtype=float)
                x_jitter = rng.normal(i, 0.045, size=len(aucs))

                ax.scatter(
                    x_jitter,
                    aucs,
                    s=18,
                    alpha=0.65,
                    edgecolor="none",
                    color=EMBEDDING_COLORS[embedding_key],
                    label=EMBEDDING_LABELS[embedding_key],
                )

        ax.axhline(
            0.5,
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.75,
        )

        ax.set_title(dataset_name, fontsize=12, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODELS],
            rotation=35,
            ha="right",
        )
        ax.set_ylim(0.45, 1.02)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("ROC-AUC", fontsize=11)
    axes[2].set_ylabel("ROC-AUC", fontsize=11)

    # Deduplicate legend handles
    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        "Architecture-dependent ROC-AUC distributions within each dataset\n"
        "points colored by embedding combination",
        fontsize=15,
        fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(
        output_dir,
        "auc_violin_by_architecture_per_dataset.png",
    )
    out_pdf = os.path.join(
        output_dir,
        "auc_violin_by_architecture_per_dataset.pdf",
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
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
    )
    parser.add_argument(
        "--balanced_dir",
        default="/p/scratch/hai_oneprot/curve_plots",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/separability_figures",
    )

    args = parser.parse_args()

    plot_architecture_violins(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()