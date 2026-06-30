#!/usr/bin/env python3
"""
plot_average_roc_improved.py

Supplementary figure: average ROC curves across all 7 OneProt encoder
architectures, 4 datasets, and 4 embedding combinations.

Updated dataset names:
    ASD_pockets_*  -> PPI-Site
    Kinase_*       -> KinSite
    merged_*       -> Dual-Site
    ASD_merged_*   -> AlloDiverse

Design:
    - 7 panels, one per encoder architecture
    - colour = dataset
    - line style = embedding combination
    - no large per-panel legends
    - AUC values saved to CSV
"""

import os
import argparse
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

TASKS = [
    "ASD_pockets_binary_comp",
    "ASD_pockets_binary_text_comp",
    "ASD_pockets_sequence_binary_comp",
    "ASD_pockets_sequence_binary_text_comp",

    "Kinase_pocket",
    "Kinase_pocket_text",
    "Kinase_combined",
    "Kinase_combined_text",

    "merged_pocket_binary_comp",
    "merged_pocket_binary_text_comp",
    "merged_pocket_sequence_binary_comp",
    "merged_pocket_sequence_binary_text_comp",

    "ASD_merged_pocket_binary_comp",
    "ASD_merged_pocket_binary_text_comp",
    "ASD_merged_pocket_sequence_binary_comp",
    "ASD_merged_pocket_sequence_binary_text_comp",
]

DATASET_LABELS = {
    "ASD_pockets": "PPI-Site",
    "Kinase": "KinSite",
    "merged": "Dual-Site",
    "ASD_merged": "AlloDiverse",
}

DATASET_COLORS = {
    "PPI-Site": "#1f77b4",
    "KinSite": "#9467bd",
    "Dual-Site": "#2ca02c",
    "AlloDiverse": "#d62728",
}

EMBEDDING_LABELS = {
    "p": "Pocket",
    "pt": "Pocket+Text",
    "ps": "Pocket+Sequence",
    "pst": "Pocket+Sequence+Text",
}

EMBEDDING_STYLES = {
    "p": "-",
    "pt": "--",
    "ps": "-.",
    "pst": ":",
}

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 250)


def task_dataset(task):
    if task.startswith("ASD_merged"):
        return "AlloDiverse"
    if task.startswith("ASD_pockets"):
        return "PPI-Site"
    if task.startswith("merged"):
        return "Dual-Site"
    if task.startswith("Kinase"):
        return "KinSite"
    return "Other"


def task_embedding(task):
    # KinSite naming
    if task == "Kinase_pocket":
        return "p"
    if task == "Kinase_pocket_text":
        return "pt"
    if task == "Kinase_combined":
        return "ps"
    if task == "Kinase_combined_text":
        return "pst"

    # ASD_pockets naming: no "pocket_" after dataset prefix
    if task.endswith("_sequence_binary_text_comp"):
        return "pst"
    if task.endswith("_sequence_binary_comp"):
        return "ps"
    if task.endswith("_binary_text_comp"):
        return "pt"
    if task.endswith("_binary_comp"):
        return "p"

    raise ValueError(f"Could not determine embedding type for task: {task}")


def load_seed_curve(base_dir, model, task, seed):
    path = os.path.join(base_dir, f"{task}_{model}_seed{seed}_preds.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr, sklearn_auc(fpr, tpr)


def average_roc(base_dir, model, task):
    tprs = []
    aucs = []

    for seed in SEEDS:
        result = load_seed_curve(base_dir, model, task, seed)
        if result is None:
            continue

        fpr, tpr, roc_auc = result
        interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    if not tprs:
        return None

    return {
        "mean_tpr": np.mean(tprs, axis=0),
        "std_tpr": np.std(tprs, axis=0),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "n": len(tprs),
    }


def plot_all(base_dir, output_dir, show_bands=True):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2, 4,
        figsize=(22, 10.5),
        sharex=True,
        sharey=True,
    )

    axes_flat = axes.flatten()
    axes_flat[-1].axis("off")

    auc_rows = []

    for i, model in enumerate(MODELS):
        ax = axes_flat[i]
        missing = []

        for task in TASKS:
            result = average_roc(base_dir, model, task)
            if result is None:
                missing.append(task)
                continue

            dataset = task_dataset(task)
            emb = task_embedding(task)

            color = DATASET_COLORS[dataset]
            if emb not in EMBEDDING_STYLES:
                raise ValueError(f"Unknown embedding key '{emb}' for task '{task}'")
            linestyle = EMBEDDING_STYLES[emb]

            label = f"{dataset} | {EMBEDDING_LABELS[emb]}"

            ax.plot(
                MEAN_FPR,
                result["mean_tpr"],
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
                alpha=0.95,
                label=label,
            )

            if show_bands:
                lower = np.clip(result["mean_tpr"] - result["std_tpr"], 0, 1)
                upper = np.clip(result["mean_tpr"] + result["std_tpr"], 0, 1)
                ax.fill_between(
                    MEAN_FPR,
                    lower,
                    upper,
                    color=color,
                    alpha=0.07,
                    linewidth=0,
                )

            auc_rows.append({
                "model": MODEL_LABELS[model],
                "model_id": model,
                "task": task,
                "dataset": dataset,
                "embedding": EMBEDDING_LABELS[emb],
                "mean_auc": result["mean_auc"],
                "std_auc": result["std_auc"],
                "n_seeds": result["n"],
            })

        ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.0)

        ax.set_title(MODEL_LABELS[model], fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.tick_params(labelsize=11)

        if i % 4 == 0:
            ax.set_ylabel("True positive rate", fontsize=12)
        if i >= 4:
            ax.set_xlabel("False positive rate", fontsize=12)

        if missing:
            print(f"[{MODEL_LABELS[model]}] Missing {len(missing)} task curves:")
            for m in missing:
                print(f"  - {m}")

    dataset_handles = [
        Line2D([0], [0], color=color, lw=3, label=label)
        for label, color in DATASET_COLORS.items()
    ]

    embedding_handles = [
        Line2D(
            [0], [0],
            color="black",
            lw=2,
            linestyle=style,
            label=EMBEDDING_LABELS[key],
        )
        for key, style in EMBEDDING_STYLES.items()
    ]

    fig.legend(
        handles=dataset_handles,
        loc="lower center",
        bbox_to_anchor=(0.33, 0.01),
        ncol=4,
        frameon=False,
        fontsize=12,
        title="Dataset",
        title_fontsize=12,
    )

    fig.legend(
        handles=embedding_handles,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.01),
        ncol=4,
        frameon=False,
        fontsize=12,
        title="Embedding combination",
        title_fontsize=12,
    )

    fig.suptitle(
        "Average ROC curves across datasets and embedding combinations",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.045,
        "Curves show seed-averaged ROC performance; shaded regions indicate ±1 SD.",
        ha="center",
        fontsize=11,
    )

    plt.tight_layout(rect=[0.02, 0.08, 1.0, 0.94])

    png_path = os.path.join(output_dir, "all_models_average_roc_improved.png")
    pdf_path = os.path.join(output_dir, "all_models_average_roc_improved.pdf")
    csv_path = os.path.join(output_dir, "all_models_average_roc_auc_values.csv")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "model_id",
                "task",
                "dataset",
                "embedding",
                "mean_auc",
                "std_auc",
                "n_seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(auc_rows)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved AUC table: {csv_path}")


def list_available_files(base_dir):
    files = [f for f in os.listdir(base_dir) if f.endswith("_preds.npz")]
    print(f"Found {len(files)} .npz files in {base_dir}")
    for f in sorted(files)[:40]:
        print(f"  {f}")
    if len(files) > 40:
        print(f"  ... and {len(files) - 40} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="<REPO_ROOT>/curve_plots",
    )
    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/average_roc_plots",
    )
    parser.add_argument(
        "--no_bands",
        action="store_true",
        help="Disable shaded ±1 SD bands.",
    )
    parser.add_argument(
        "--list_files",
        action="store_true",
        help="Print available .npz files and exit.",
    )

    args = parser.parse_args()

    if args.list_files:
        list_available_files(args.base_dir)
        return

    print(f"Reading from: {args.base_dir}")
    print(f"Writing to:   {args.output_dir}")

    plot_all(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        show_bands=not args.no_bands,
    )

    print("Done.")


if __name__ == "__main__":
    main()