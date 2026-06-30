#!/usr/bin/env python3
"""
plot_auc_violin_by_dataset_all_embeddings.py

Composite 2x2 ROC-AUC violin figure.

Usage:
python plot_auc_violin_by_dataset_all_embeddings.py \
    --base_dir <REPO_ROOT>/curve_plots \
    --balanced_dir <CURVE_PLOTS_ROOT> \
    --output_dir <REPO_ROOT>/separability_figures
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

DATASET_LABELS = {
    "PL8": "PPI-Site",
    "Kinase": "KinSite",
    "PL8 + Kinase": "Dual-Site",
    "ASD + PL8 + Kinase": "AlloDiverse",
}

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

    if "y_true" not in data or "y_pred" not in data:
        raise KeyError(
            f"{path} does not contain y_true and y_pred. "
            f"Available keys: {list(data.keys())}"
        )

    y_true = np.asarray(data["y_true"]).astype(int)
    y_pred = np.asarray(data["y_pred"])

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


def collect_all_auc_values(base_dir, balanced_dir):
    rows = []
    missing = []

    for embedding_key in EMBEDDING_ORDER:
        for dataset_name in DATASET_ORDER:
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

                    if auc_value is None:
                        missing.append(
                            {
                                "embedding": embedding_key,
                                "dataset": dataset_name,
                                "model": model,
                                "seed": seed,
                                "path": path,
                            }
                        )
                        continue

                    rows.append(
                        {
                            "embedding": embedding_key,
                            "embedding_label": EMBEDDING_LABELS[embedding_key],
                            "dataset": dataset_name,
                            "dataset_label": DATASET_LABELS[dataset_name],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "seed": seed,
                            "auc": auc_value,
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(missing)


def draw_panel(ax, df, embedding_key, panel_label):
    sub = df[df["embedding"] == embedding_key]
    color = EMBEDDING_COLORS[embedding_key]

    data_by_dataset = []

    for dataset_name in DATASET_ORDER:
        values = sub[sub["dataset"] == dataset_name]["auc"].to_numpy(dtype=float)

        if len(values) == 0:
            raise RuntimeError(
                f"No AUC values for embedding={embedding_key}, dataset={dataset_name}"
            )

        data_by_dataset.append(values)

    positions = np.arange(1, len(DATASET_ORDER) + 1)

    violin_parts = ax.violinplot(
        data_by_dataset,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.72,
    )

    for body in violin_parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(1.1)

    rng = np.random.default_rng(123)

    for i, values in enumerate(data_by_dataset, start=1):
        x_jitter = rng.normal(i, 0.055, size=len(values))

        ax.scatter(
            x_jitter,
            values,
            s=22,
            alpha=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        mean = np.mean(values)

        ax.plot(
            [i - 0.22, i + 0.22],
            [median, median],
            color="black",
            linewidth=2.0,
            zorder=4,
        )

        ax.plot(
            [i, i],
            [q1, q3],
            color="black",
            linewidth=3.0,
            alpha=0.85,
            zorder=4,
        )

        ax.scatter(
            [i],
            [mean],
            s=42,
            color="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.65,
        zorder=1,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [DATASET_LABELS[d] for d in DATASET_ORDER],
        rotation=15,
        ha="right",
    )

    ax.set_ylim(0.43, 1.02)
    ax.set_xlim(0.45, len(DATASET_ORDER) + 0.55)

    ax.grid(axis="y", alpha=0.22, linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(
        f"{panel_label}. {EMBEDDING_LABELS[embedding_key]}",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )


def plot_composite(df, output_dir):
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.75, 11.55),
        sharey=True,
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.86,
        bottom=0.14,
        hspace=0.35,
        wspace=0.10,
    )

    axes = axes.flatten()
    panel_labels = ["A", "B", "C", "D"]

    for ax, embedding_key, panel_label in zip(
        axes,
        EMBEDDING_ORDER,
        panel_labels,
    ):
        draw_panel(
            ax=ax,
            df=df,
            embedding_key=embedding_key,
            panel_label=panel_label,
        )

    axes[0].set_ylabel("ROC-AUC")
    axes[2].set_ylabel("ROC-AUC")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.0,
            label="Median",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Mean",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=3.0,
            alpha=0.85,
            label="IQR",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="gray",
            markeredgecolor="white",
            label="Individual run",
        ),
    ]

    fig.suptitle(
        "Dataset-dependent separability regimes across embedding combinations",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=14,
        handlelength=1.5,
        columnspacing=1.4,
    )

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(
        output_dir,
        "auc_violin_by_dataset_all_embeddings_composite.png",
    )
    out_pdf = os.path.join(
        output_dir,
        "auc_violin_by_dataset_all_embeddings_composite.pdf",
    )
    out_csv = os.path.join(
        output_dir,
        "auc_violin_by_dataset_all_embeddings_values.csv",
    )
    out_summary = os.path.join(
        output_dir,
        "auc_violin_by_dataset_all_embeddings_summary.csv",
    )

    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["embedding_label", "dataset_label"], as_index=False)
        .agg(
            n=("auc", "count"),
            mean_auc=("auc", "mean"),
            median_auc=("auc", "median"),
            sd_auc=("auc", "std"),
            min_auc=("auc", "min"),
            max_auc=("auc", "max"),
        )
    )
    summary.to_csv(out_summary, index=False)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG -> {out_png}")
    print(f"Saved PDF -> {out_pdf}")
    print(f"Saved values -> {out_csv}")
    print(f"Saved summary -> {out_summary}")


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

    df, missing = collect_all_auc_values(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
    )

    if df.empty:
        raise RuntimeError("No AUC values loaded. Check paths.")

    os.makedirs(args.output_dir, exist_ok=True)

    missing_csv = os.path.join(
        args.output_dir,
        "auc_violin_by_dataset_all_embeddings_missing.csv",
    )
    missing.to_csv(missing_csv, index=False)

    print(f"Loaded {len(df)} AUC values.")
    print(f"Missing/skipped entries: {len(missing)}")
    print(f"Saved missing report -> {missing_csv}")

    plot_composite(df=df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()