#!/usr/bin/env python3
"""
plot_tpr_tnr_boxplots_supplement_with_excel_no_seed.py

Supplementary class-specific performance figure:
    - 2 rows:
        top    = TPR / sensitivity / true-positive rate
        bottom = TNR / specificity / true-negative rate
    - 4 columns = datasets
    - boxplots summarize values across encoder architectures and runs
    - overlaid points show individual runs, colored by embedding combination

This version:
    - uses seeds internally to find files
    - does NOT write seed values to CSV or Excel
    - exports a source-data Excel file for the supplementary figure

Usage:
    python plot_tpr_tnr_boxplots_supplement_with_excel_no_seed.py

Optional:
    python plot_tpr_tnr_boxplots_supplement_with_excel_no_seed.py \
        --base_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots \
        --balanced_dir /p/scratch/hai_oneprot/curve_plots \
        --output_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/separability_figures
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
    "PL8 + Kinase": "DualSite",
    "ASD + PL8 + Kinase": "AlloSite",
}

DATASET_COLORS = {
    "PL8": "#1f77b4",
    "Kinase": "#9467bd",
    "PL8 + Kinase": "#2ca02c",
    "ASD + PL8 + Kinase": "#d62728",
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


def load_predictions(path):
    if not os.path.exists(path):
        return None, None

    data = np.load(path)

    if "y_true" not in data or "y_pred" not in data:
        raise KeyError(
            f"{path} does not contain y_true and y_pred. "
            f"Available keys: {list(data.keys())}"
        )

    y_true = np.asarray(data["y_true"]).astype(int).reshape(-1)
    y_score = np.asarray(data["y_pred"]).reshape(-1)

    return y_true, y_score


def compute_tpr_tnr(y_true, y_score, threshold=0.5):
    y_hat = (y_score >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_hat == 1))
    tn = np.sum((y_true == 0) & (y_hat == 0))
    fp = np.sum((y_true == 0) & (y_hat == 1))
    fn = np.sum((y_true == 1) & (y_hat == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return tpr, tnr


def collect_values(base_dir, balanced_dir, threshold):
    rows = []
    missing = []

    for dataset_name in DATASET_ORDER:
        for model in MODELS:
            for embedding_key in EMBEDDING_ORDER:
                for seed in SEEDS:
                    path = get_path(
                        base_dir=base_dir,
                        balanced_dir=balanced_dir,
                        dataset_name=dataset_name,
                        embedding_key=embedding_key,
                        model=model,
                        seed=seed,
                    )

                    y_true, y_score = load_predictions(path)

                    if y_true is None:
                        missing.append(
                            {
                                "dataset": DATASET_LABELS[dataset_name],
                                "embedding": EMBEDDING_LABELS[embedding_key],
                                "architecture": MODEL_LABELS[model],
                                "path": path,
                            }
                        )
                        continue

                    if len(np.unique(y_true)) < 2:
                        missing.append(
                            {
                                "dataset": DATASET_LABELS[dataset_name],
                                "embedding": EMBEDDING_LABELS[embedding_key],
                                "architecture": MODEL_LABELS[model],
                                "path": path,
                            }
                        )
                        continue

                    tpr, tnr = compute_tpr_tnr(
                        y_true=y_true,
                        y_score=y_score,
                        threshold=threshold,
                    )

                    rows.append(
                        {
                            "dataset_key": dataset_name,
                            "dataset": DATASET_LABELS[dataset_name],
                            "embedding_key": embedding_key,
                            "embedding": EMBEDDING_LABELS[embedding_key],
                            "architecture_key": model,
                            "architecture": MODEL_LABELS[model],
                            "tpr": tpr,
                            "tnr": tnr,
                        }
                    )

    return rows, missing


def draw_boxplot_with_points(ax, rows, dataset_name, metric, ylabel=None):
    rng = np.random.default_rng(123)

    data_by_embedding = []

    for embedding_key in EMBEDDING_ORDER:
        values = [
            r[metric]
            for r in rows
            if r["dataset_key"] == dataset_name
            and r["embedding_key"] == embedding_key
            and not np.isnan(r[metric])
        ]

        data_by_embedding.append(np.array(values, dtype=float))

    positions = np.arange(1, len(EMBEDDING_ORDER) + 1)

    bp = ax.boxplot(
        data_by_embedding,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": "black",
            "linewidth": 1.5,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.0,
        },
        capprops={
            "color": "black",
            "linewidth": 1.0,
        },
        boxprops={
            "linewidth": 1.0,
            "color": "black",
        },
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(DATASET_COLORS[dataset_name])
        patch.set_alpha(0.25)

    for i, embedding_key in enumerate(EMBEDDING_ORDER, start=1):
        values = [
            r[metric]
            for r in rows
            if r["dataset_key"] == dataset_name
            and r["embedding_key"] == embedding_key
            and not np.isnan(r[metric])
        ]

        if not values:
            continue

        values = np.array(values, dtype=float)
        x_jitter = rng.normal(i, 0.055, size=len(values))

        ax.scatter(
            x_jitter,
            values,
            s=22,
            alpha=0.70,
            edgecolor="white",
            linewidth=0.25,
            color=EMBEDDING_COLORS[embedding_key],
            zorder=3,
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.55,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER],
        rotation=28,
        ha="right",
        fontsize=10,
    )

    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", alpha=0.25)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)

    ax.tick_params(axis="y", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def export_source_data(rows, missing, output_dir, threshold):
    df = pd.DataFrame(rows)

    source_df = df[
        [
            "dataset",
            "embedding",
            "architecture",
            "tpr",
            "tnr",
        ]
    ].copy()

    source_df = source_df.rename(
        columns={
            "dataset": "Dataset",
            "embedding": "Embedding",
            "architecture": "Architecture",
            "tpr": "TPR",
            "tnr": "TNR",
        }
    )

    summary = (
        source_df.groupby(["Dataset", "Embedding"], as_index=False)
        .agg(
            N=("TPR", "count"),
            Mean_TPR=("TPR", "mean"),
            SD_TPR=("TPR", "std"),
            Median_TPR=("TPR", "median"),
            Min_TPR=("TPR", "min"),
            Max_TPR=("TPR", "max"),
            Mean_TNR=("TNR", "mean"),
            SD_TNR=("TNR", "std"),
            Median_TNR=("TNR", "median"),
            Min_TNR=("TNR", "min"),
            Max_TNR=("TNR", "max"),
        )
    )

    metadata = pd.DataFrame(
        [
            {
                "Field": "Figure",
                "Value": "Supplementary TPR/TNR boxplots",
            },
            {
                "Field": "Metric",
                "Value": "TPR and TNR computed at fixed classification threshold",
            },
            {
                "Field": "Threshold",
                "Value": threshold,
            },
            {
                "Field": "Seed column",
                "Value": "Not exported; seeds were used only internally to locate replicate prediction files.",
            },
            {
                "Field": "Rows",
                "Value": "Each row corresponds to one individual run used in the boxplots.",
            },
        ]
    )

    missing_df = pd.DataFrame(missing)

    csv_path = os.path.join(
        output_dir,
        "supplement_tpr_tnr_boxplot_values_no_seed.csv",
    )

    excel_path = os.path.join(
        output_dir,
        "Supplementary_TPR_TNR_SourceData_no_seed.xlsx",
    )

    source_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        source_df.to_excel(writer, sheet_name="Raw_Data", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

        if not missing_df.empty:
            missing_df.to_excel(writer, sheet_name="Missing", index=False)

    print(f"Saved source CSV -> {csv_path}")
    print(f"Saved source Excel -> {excel_path}")


def plot_figure(rows, output_dir):
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(19, 9.5),
        sharey=True,
    )

    for col, dataset_name in enumerate(DATASET_ORDER):
        axes[0, col].set_title(
            DATASET_LABELS[dataset_name],
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        draw_boxplot_with_points(
            ax=axes[0, col],
            rows=rows,
            dataset_name=dataset_name,
            metric="tpr",
            ylabel="TPR / sensitivity" if col == 0 else None,
        )

        draw_boxplot_with_points(
            ax=axes[1, col],
            rows=rows,
            dataset_name=dataset_name,
            metric="tnr",
            ylabel="TNR / specificity" if col == 0 else None,
        )

    axes[0, 0].text(
        -0.42,
        0.5,
        "Allosteric sites",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    axes[1, 0].text(
        -0.42,
        0.5,
        "Orthosteric sites",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=EMBEDDING_COLORS[e],
            markeredgecolor=EMBEDDING_COLORS[e],
            label=EMBEDDING_LABELS[e],
            markersize=8,
        )
        for e in EMBEDDING_ORDER
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        "Threshold-dependent class-specific performance across datasets",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.025,
        "Boxes show median and interquartile range across encoder architectures and runs; "
        "points show individual runs colored by embedding combination.",
        ha="center",
        va="center",
        fontsize=10,
    )

    plt.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.875,
        bottom=0.15,
        wspace=0.18,
        hspace=0.42,
    )

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(output_dir, "supplement_tpr_tnr_boxplots.png")
    out_pdf = os.path.join(output_dir, "supplement_tpr_tnr_boxplots.pdf")

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
        help="Directory containing flat prediction files.",
    )

    parser.add_argument(
        "--balanced_dir",
        default="/p/scratch/hai_oneprot/curve_plots",
        help="Directory containing balanced ASD prediction folders.",
    )

    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/separability_figures",
        help="Directory for output figures and source-data files.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold used to compute TPR and TNR.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows, missing = collect_values(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        threshold=args.threshold,
    )

    if not rows:
        raise RuntimeError("No values loaded. Check input paths.")

    print(f"Loaded values: {len(rows)}")
    print(f"Missing/skipped entries: {len(missing)}")

    export_source_data(
        rows=rows,
        missing=missing,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )

    plot_figure(
        rows=rows,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()