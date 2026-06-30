#!/usr/bin/env python3
"""
plot_all_models_average_roc_publication_style.py

Publication-style supplementary ROC figure for all 7 OneProt encoder
architectures across four allosteric-site regimes and four downstream
embedding combinations.

Designed to match the visual style of the cleaned representative/architecture
ROC figures:
  - large bold sans-serif fonts
  - grid-free panels
  - square ROC panels
  - separate dataset titles and clean panel labels
  - no legends inside ROC panels
  - shared legend/info area in the empty 8th panel of a 4 x 2 layout
  - high-resolution outputs: PDF, SVG, PNG, TIFF
  - AUC values exported to CSV

Layout:
  - 4 rows x 2 columns
  - first 7 panels = model/encoder architectures
  - final bottom-right panel = shared legend and shaded-band explanation
  - reduced figure/panel title font sizes to avoid crowding
  - enlarged legend fonts for readability

Outputs:
  all_models_average_roc_publication_style.pdf
  all_models_average_roc_publication_style.svg
  all_models_average_roc_publication_style.png
  all_models_average_roc_publication_style.tif
  all_models_average_roc_auc_values.csv
"""

import os
import csv
import argparse
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, auc as sklearn_auc


# -----------------------------------------------------------------------------
# Publication-style matplotlib settings
# -----------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "axes.linewidth": 1.8,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",

    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,

    "lines.linewidth": 2.6,
    "legend.fontsize": 16,
    "legend.frameon": False,

    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODELS = [
    "oneprot_pocket_text_32900",
    "oneprot_struct_token_pocket_text_32900",
    "oneprot_struct_graph_pocket_text_32900",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity",
    "oneprot_md_combined_gpcr_no_struct_graph_32900",
    "oneprot_md_combined_gpcr_no_struct_token_32900",
    "oneprot_md_combined_gpcr_32900",
]

MODEL_LABELS = {
    "oneprot_pocket_text_32900": "Pocket+Text",
    "oneprot_struct_token_pocket_text_32900": "ST+Pocket+Text",
    "oneprot_struct_graph_pocket_text_32900": "SG+Pocket+Text",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "ST+SG+Pocket+Text",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "MD+ST+Pocket+Text",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "MD+SG+Pocket+Text",
    "oneprot_md_combined_gpcr_32900": "MD+ST+SG+Pocket+Text",
}

PANEL_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

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

DATASET_ORDER = ["PPI-Site", "KinSite", "Dual-Site", "AlloDiverse"]

DATASET_COLORS = {
    "PPI-Site": "#0072B2",
    "KinSite": "#CC79A7",
    "Dual-Site": "#009E73",
    "AlloDiverse": "#D55E00",
}

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket only",
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
MEAN_FPR = np.linspace(0, 1, 600)

TITLE_COLOR = "#111111"
REGIME_COLOR = "#0B3D91"


# -----------------------------------------------------------------------------
# Task helpers
# -----------------------------------------------------------------------------

def task_dataset(task: str) -> str:
    if task.startswith("ASD_merged"):
        return "AlloDiverse"
    if task.startswith("ASD_pockets"):
        return "PPI-Site"
    if task.startswith("merged"):
        return "Dual-Site"
    if task.startswith("Kinase"):
        return "KinSite"
    raise ValueError(f"Could not determine dataset for task: {task}")


def task_embedding(task: str) -> str:
    # KinSite naming
    if task == "Kinase_pocket":
        return "p"
    if task == "Kinase_pocket_text":
        return "pt"
    if task == "Kinase_combined":
        return "ps"
    if task == "Kinase_combined_text":
        return "pst"

    # PPI-Site / Dual-Site / AlloDiverse naming
    if task.endswith("_sequence_binary_text_comp"):
        return "pst"
    if task.endswith("_sequence_binary_comp"):
        return "ps"
    if task.endswith("_binary_text_comp"):
        return "pt"
    if task.endswith("_binary_comp"):
        return "p"

    raise ValueError(f"Could not determine embedding type for task: {task}")


# -----------------------------------------------------------------------------
# Loading and averaging
# -----------------------------------------------------------------------------

def _load_npz(path: str):
    if not os.path.exists(path):
        return None

    try:
        data = np.load(path)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = sklearn_auc(fpr, tpr)
    return fpr, tpr, roc_auc


def load_seed_curve(base_dir: str, model: str, task: str, seed: int):
    path = os.path.join(base_dir, f"{task}_{model}_seed{seed}_preds.npz")
    return _load_npz(path)


def average_roc(base_dir: str, model: str, task: str):
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

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    return {
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "n": len(tprs),
    }


# -----------------------------------------------------------------------------
# Plot styling helpers
# -----------------------------------------------------------------------------

def bold_tick_labels(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def style_axis(ax, show_xlabel=True, show_ylabel=True):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")

    if show_xlabel:
        ax.set_xlabel("False positive rate", fontsize=18, fontweight="bold", labelpad=8)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("True positive rate", fontsize=18, fontweight="bold", labelpad=8)
    else:
        ax.set_ylabel("")

    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="both", direction="out", width=1.6, length=6)
    bold_tick_labels(ax)

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)


def add_panel_label(ax, label):
    ax.text(
        -0.095,
        1.15,
        label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def add_panel_title(ax, title: str):
    ax.text(
        0.5,
        1.050,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=17.5,
        fontweight="bold",
        color=TITLE_COLOR,
    )


def make_dataset_handles():
    return [
        Line2D([0], [0], color=DATASET_COLORS[label], lw=4.0, label=label)
        for label in DATASET_ORDER
    ]


def make_embedding_handles():
    return [
        Line2D(
            [0], [0],
            color="black",
            lw=3.0,
            linestyle=EMBEDDING_STYLES[key],
            label=EMBEDDING_LABELS[key],
        )
        for key in EMBEDDING_ORDER
    ]


def draw_legend_panel(legend_ax, show_bands: bool):
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    legend_ax.text(
        0.50,
        0.96,
        "Shared encoding",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=TITLE_COLOR,
    )

    legend_ax.text(
        0.06,
        0.82,
        "Dataset",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="black",
    )
    dataset_legend = legend_ax.legend(
        handles=make_dataset_handles(),
        loc="upper left",
        bbox_to_anchor=(0.04, 0.75),
        fontsize=16,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.85,
        labelspacing=0.72,
        borderaxespad=0.0,
    )
    legend_ax.add_artist(dataset_legend)
    for text in dataset_legend.get_texts():
        text.set_fontweight("bold")

    legend_ax.text(
        0.54,
        0.82,
        "Embedding",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="black",
    )
    embedding_legend = legend_ax.legend(
        handles=make_embedding_handles(),
        loc="upper left",
        bbox_to_anchor=(0.52, 0.75),
        fontsize=16,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.85,
        labelspacing=0.72,
        borderaxespad=0.0,
    )
    for text in embedding_legend.get_texts():
        text.set_fontweight("bold")

    legend_ax.plot([0.05, 0.95], [0.30, 0.30], color="black", lw=1.1)

    legend_ax.add_patch(
        Rectangle(
            (0.08, 0.18),
            0.09,
            0.06,
            facecolor="#BFC7D5",
            edgecolor="none",
            alpha=0.65,
            clip_on=False,
            zorder=5,
        )
    )
    note = "Shaded regions indicate ±1 standard deviation across runs" if show_bands else "Shaded regions disabled"
    legend_ax.text(
        0.22,
        0.21,
        note,
        ha="left",
        va="center",
        fontsize=16.5,
        color="black",
        linespacing=1.18,
    )

    legend_ax.plot(
        [0.08, 0.18],
        [0.08, 0.08],
        color="black",
        lw=2.0,
        linestyle="--",
    )
    legend_ax.text(
        0.22,
        0.08,
        "Random classifier",
        ha="left",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color="black",
    )


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------

def plot_all(base_dir: str, output_dir: str, show_bands: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(14.5, 22.0), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        wspace=0.10,
        hspace=0.34,
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(2)]
    legend_ax = axes[-1]
    roc_axes = axes[:-1]

    auc_rows = []

    for i, (model, ax) in enumerate(zip(MODELS, roc_axes)):
        missing = []

        for task in TASKS:
            result = average_roc(base_dir, model, task)
            if result is None:
                missing.append(task)
                continue

            dataset = task_dataset(task)
            emb = task_embedding(task)

            color = DATASET_COLORS[dataset]
            linestyle = EMBEDDING_STYLES[emb]

            ax.plot(
                MEAN_FPR,
                result["mean_tpr"],
                color=color,
                linestyle=linestyle,
                linewidth=2.45,
                alpha=0.96,
                solid_capstyle="round",
                zorder=3,
            )

            if show_bands:
                ax.fill_between(
                    MEAN_FPR,
                    np.clip(result["mean_tpr"] - result["std_tpr"], 0, 1),
                    np.clip(result["mean_tpr"] + result["std_tpr"], 0, 1),
                    color=color,
                    alpha=0.065,
                    linewidth=0,
                    zorder=2,
                )

            auc_rows.append({
                "model": MODEL_LABELS[model],
                "model_id": model,
                "task": task,
                "dataset": dataset,
                "embedding": EMBEDDING_LABELS[emb],
                "mean_auc": result["mean_auc"],
                "std_auc": result["std_auc"],
                "n_runs": result["n"],
            })

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            lw=2.0,
            alpha=0.75,
            zorder=1,
        )

        add_panel_label(ax, PANEL_LABELS[i])
        add_panel_title(ax, MODEL_LABELS[model])

        row = i // 2
        col = i % 2
        style_axis(
            ax,
            show_xlabel=row == 3 or i == 6,
            show_ylabel=col == 0,
        )

        # Square axes can leave visual gaps inside GridSpec cells; this gently
        # pulls the two columns inward while keeping a small gap.
        if col == 0:
            ax.set_anchor((0.68, 0.50))
        else:
            ax.set_anchor((0.32, 0.50))

        if missing:
            print(f"[{MODEL_LABELS[model]}] Missing {len(missing)} task curves:")
            for task in missing:
                print(f"  - {task}")

    draw_legend_panel(legend_ax, show_bands=show_bands)

    fig.suptitle(
        "Average ROC curves across datasets and embedding combinations",
        fontsize=21,
        fontweight="bold",
        y=0.992,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.965,
        bottom=0.035,
        top=0.950,
        wspace=0.10,
        hspace=0.34,
    )

    stem = os.path.join(output_dir, "all_models_average_roc_publication_style")
    png_path = f"{stem}.png"
    pdf_path = f"{stem}.pdf"
    svg_path = f"{stem}.svg"
    tif_path = f"{stem}.tif"
    csv_path = os.path.join(output_dir, "all_models_average_roc_auc_values.csv")

    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=600)
    fig.savefig(tif_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
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
                "n_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(auc_rows)

    print(f"Saved PDF  -> {pdf_path}")
    print(f"Saved SVG  -> {svg_path}")
    print(f"Saved PNG  -> {png_path}")
    print(f"Saved TIFF -> {tif_path}")
    print(f"Saved CSV  -> {csv_path}")


# -----------------------------------------------------------------------------
# Utilities and CLI
# -----------------------------------------------------------------------------

def list_available_files(base_dir: str):
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
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
        help="Directory with flat-format prediction .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/average_roc_plots",
        help="Output directory.",
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
