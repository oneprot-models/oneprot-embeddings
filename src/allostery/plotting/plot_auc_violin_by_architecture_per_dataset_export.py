#!/usr/bin/env python3
"""
plot_auc_violin_by_architecture_per_dataset_colored_no_seed_excel.py
"""

import os
import argparse
import numpy as np
import pandas as pd

# Excel output requires openpyxl or xlsxwriter in the Python environment.
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

DATASET_COLORS = {
    "PL8": "#1f77b4",
    "Kinase": "#ff7f0e",
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


def collect_auc_values(base_dir, balanced_dir):
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

                    auc_value = load_auc(path)

                    if auc_value is None:
                        missing.append(
                            {
                                "dataset": dataset_name,
                                "dataset_label": DATASET_LABELS[dataset_name],
                                "model": model,
                                "model_label": MODEL_LABELS[model],
                                "embedding": embedding_key,
                                "embedding_label": EMBEDDING_LABELS[embedding_key],
                                "path": path,
                            }
                        )
                        continue

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "dataset_label": DATASET_LABELS[dataset_name],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "embedding": embedding_key,
                            "embedding_label": EMBEDDING_LABELS[embedding_key],
                            "auc": auc_value,
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(missing)


def draw_panel(ax, df, dataset_name, panel_label):
    sub = df[df["dataset"] == dataset_name]
    positions = np.arange(1, len(MODELS) + 1)
    color = DATASET_COLORS[dataset_name]

    data_by_model = []

    for model in MODELS:
        values = sub[sub["model"] == model]["auc"].to_numpy(dtype=float)

        if len(values) == 0:
            raise RuntimeError(
                f"No AUC values for dataset={dataset_name}, model={model}"
            )

        data_by_model.append(values)

    violin_parts = ax.violinplot(
        data_by_model,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.72,
    )

    for body in violin_parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.50)
        body.set_linewidth(1.5)

    rng = np.random.default_rng(123)

    for i, values in enumerate(data_by_model, start=1):
        x_jitter = rng.normal(i, 0.045, size=len(values))

        ax.scatter(
            x_jitter,
            values,
            s=28,
            alpha=0.82,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )

        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        mean = np.mean(values)

        ax.plot(
            [i - 0.23, i + 0.23],
            [median, median],
            color="black",
            linewidth=2.4,
            zorder=6,
        )

        ax.plot(
            [i, i],
            [q1, q3],
            color="black",
            linewidth=3.5,
            alpha=0.9,
            zorder=5,
        )

        ax.scatter(
            [i],
            [mean],
            s=58,
            color="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=7,
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.1,
        color="black",
        alpha=0.65,
        zorder=1,
    )

    ax.set_title(
        f"{panel_label}. {DATASET_LABELS[dataset_name]}",
        fontsize=20,
        fontweight="bold",
        pad=14,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [MODEL_LABELS[m] for m in MODELS],
        rotation=35,
        ha="right",
        fontsize=14.5,
    )

    ax.set_ylim(0.40, 1.02)
    ax.set_xlim(0.45, len(MODELS) + 0.55)

    ax.grid(axis="y", alpha=0.22, linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_architecture_violins(base_dir, balanced_dir, output_dir):
    df, missing = collect_auc_values(base_dir, balanced_dir)

    if df.empty:
        raise RuntimeError("No AUC values loaded. Check input directories.")

    plt.rcParams.update(
        {
            "font.size": 19,
            "axes.titlesize": 20,
            "axes.labelsize": 22,
            "xtick.labelsize": 16.5,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(19.0, 14.0),
        sharey=True,
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.86,
        bottom=0.27,
        hspace=0.80,
        wspace=0.08,
    )

    axes = axes.flatten()
    panel_labels = ["A", "B", "C", "D"]

    for ax, dataset_name, panel_label in zip(
        axes,
        DATASET_ORDER,
        panel_labels,
    ):
        draw_panel(
            ax=ax,
            df=df,
            dataset_name=dataset_name,
            panel_label=panel_label,
        )

    axes[0].set_ylabel("ROC-AUC")
    axes[2].set_ylabel("ROC-AUC")

    dataset_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=DATASET_COLORS[d],
            markeredgecolor=DATASET_COLORS[d],
            label=DATASET_LABELS[d],
            markersize=9,
        )
        for d in DATASET_ORDER
    ]

    statistic_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.4,
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
            markersize=8.5,
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=3.5,
            alpha=0.9,
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
            markersize=8.5,
        ),
    ]

    fig.suptitle(
        "Architecture-dependent ROC-AUC distributions within each dataset",
        fontsize=25,
        fontweight="bold",
        y=0.975,
    )

    dataset_legend = fig.legend(
        handles=dataset_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.32, 0.055),
        fontsize=18,
        handlelength=1.4,
        columnspacing=1.5,
        title="Dataset",
        title_fontsize=20,
    )

    fig.add_artist(dataset_legend)

    fig.legend(
        handles=statistic_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.73, 0.055),
        fontsize=14,
        handlelength=1.6,
        columnspacing=1.5,
        title="Summary statistics",
        title_fontsize=20,
    )

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(
        output_dir,
        "auc_violin_by_architecture_per_dataset_colored.png",
    )
    out_pdf = os.path.join(
        output_dir,
        "auc_violin_by_architecture_per_dataset_colored.pdf",
    )
    out_csv = os.path.join(
        output_dir,
        "auc_violin_by_architecture_per_dataset_values.csv",
    )
    out_summary = os.path.join(
        output_dir,
        "auc_violin_by_architecture_per_dataset_summary.csv",
    )
    out_missing = os.path.join(
        output_dir,
        "auc_violin_by_architecture_per_dataset_missing.csv",
    )
    out_xlsx = os.path.join(
        output_dir,
        "figure7_auc_violin_source_data_no_seed.xlsx",
    )

    df.to_csv(out_csv, index=False)
    missing.to_csv(out_missing, index=False)

    summary = (
        df.groupby(["dataset_label", "model_label"], as_index=False)
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

    # One workbook containing everything needed to recreate Figure 7.
    # Raw_AUC is the key sheet: one row = one dot in the violin plot. Seed identifiers are intentionally not written.
    by_dataset_architecture = (
        df.groupby(["dataset_label", "dataset", "model_label", "model"], as_index=False)
        .agg(
            n=("auc", "count"),
            mean_auc=("auc", "mean"),
            median_auc=("auc", "median"),
            q1_auc=("auc", lambda x: x.quantile(0.25)),
            q3_auc=("auc", lambda x: x.quantile(0.75)),
            sd_auc=("auc", "std"),
            min_auc=("auc", "min"),
            max_auc=("auc", "max"),
        )
    )

    by_dataset_architecture_embedding = (
        df.groupby(
            [
                "dataset_label",
                "dataset",
                "model_label",
                "model",
                "embedding_label",
                "embedding",
            ],
            as_index=False,
        )
        .agg(
            n=("auc", "count"),
            mean_auc=("auc", "mean"),
            median_auc=("auc", "median"),
            sd_auc=("auc", "std"),
            min_auc=("auc", "min"),
            max_auc=("auc", "max"),
        )
    )

    metadata = pd.DataFrame(
        [
            {"field": "figure", "value": "Figure 7"},
            {"field": "metric", "value": "ROC-AUC"},
            {"field": "random_performance_line", "value": 0.5},
            {"field": "datasets", "value": ", ".join(DATASET_LABELS[d] for d in DATASET_ORDER)},
            {"field": "embedding_order", "value": ", ".join(EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER)},
            {"field": "model_order", "value": ", ".join(MODEL_LABELS[m] for m in MODELS)},
            {"field": "replicates_per_configuration", "value": len(SEEDS)},
            {"field": "base_dir", "value": base_dir},
            {"field": "balanced_dir", "value": balanced_dir},
        ]
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw_AUC", index=False)
        by_dataset_architecture.to_excel(
            writer, sheet_name="By_Dataset_Architecture", index=False
        )
        by_dataset_architecture_embedding.to_excel(
            writer, sheet_name="By_Dataset_Arch_Embedding", index=False
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)
        missing.to_excel(writer, sheet_name="Missing", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 42)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded AUC values: {len(df)}")
    print(f"Missing/skipped entries: {len(missing)}")
    print(f"Saved PNG -> {out_png}")
    print(f"Saved PDF -> {out_pdf}")
    print(f"Saved values -> {out_csv}")
    print(f"Saved summary -> {out_summary}")
    print(f"Saved missing report -> {out_missing}")
    print(f"Saved Excel source data -> {out_xlsx}")


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

    plot_architecture_violins(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()