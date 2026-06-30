#!/usr/bin/env python3
"""
plot_supplement_all_models_pr_4x2_publication_style.py

Publication-style supplementary precision-recall figure for all 7 OneProt
encoder architectures.

Designed to match the 4 x 2 publication-style ROC layout:
  - 4 rows x 2 columns
  - first 7 panels = model/encoder architectures
  - final bottom-right panel = shared legend and shaded-band explanation
  - large bold sans-serif fonts
  - grid-free panels
  - no legends inside PR panels
  - shared legend/info area in the empty 8th panel
  - high-resolution outputs: PDF, SVG, PNG, TIFF

Curves:
  - color = extracted downstream embedding combination
  - line style = dataset/training condition
      solid  = AlloDiverse, original training
      dashed = AlloDiverse, balanced training
      dotted = KinSite

Outputs:
  supplement_all_models_pr_4x2_publication_style.pdf
  supplement_all_models_pr_4x2_publication_style.svg
  supplement_all_models_pr_4x2_publication_style.png
  supplement_all_models_pr_4x2_publication_style.tif
  supplement_all_models_pr_aupr_values.csv
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
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc


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

    "lines.linewidth": 2.8,
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
    ("oneprot_pocket_text_32900", "Pocket+Text"),
    ("oneprot_struct_token_pocket_text_32900", "ST+Pocket+Text"),
    ("oneprot_struct_graph_pocket_text_32900", "SG+Pocket+Text"),
    ("oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity", "ST+SG+Pocket+Text"),
    ("oneprot_md_combined_gpcr_no_struct_graph_32900", "MD+ST+Pocket+Text"),
    ("oneprot_md_combined_gpcr_no_struct_token_32900", "MD+SG+Pocket+Text"),
    ("oneprot_md_combined_gpcr_32900", "MD+ST+SG+Pocket+Text"),
]

PANEL_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

SEEDS = [11, 12, 13, 14, 15]
MEAN_RECALL = np.linspace(0, 1, 600)

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket only",
    "pt": "Pocket+Text",
    "ps": "Pocket+Sequence",
    "pst": "Pocket+Sequence+Text",
}

# Same color scheme as the representative embedding-level ROC figure.
EMBEDDING_COLORS = {
    "p": "#0072B2",
    "pt": "#E69F00",
    "ps": "#009E73",
    "pst": "#D55E00",
}

CONDITION_ORDER = ["allodiverse_original", "allodiverse_balanced", "kinsite"]

CONDITION_LABELS = {
    "allodiverse_original": "AlloDiverse, original training",
    "allodiverse_balanced": "AlloDiverse, balanced training",
    "kinsite": "KinSite",
}

CONDITION_STYLES = {
    "allodiverse_original": "-",
    "allodiverse_balanced": "--",
    "kinsite": ":",
}

ALLODIVERSE_TASKS = {
    "p": "ASD_merged_pocket_binary_comp",
    "pt": "ASD_merged_pocket_binary_text_comp",
    "ps": "ASD_merged_pocket_sequence_binary_comp",
    "pst": "ASD_merged_pocket_sequence_binary_text_comp",
}

KINSITE_TASKS = {
    "p": "Kinase_pocket",
    "pt": "Kinase_pocket_text",
    "ps": "Kinase_combined",
    "pst": "Kinase_combined_text",
}

TITLE_COLOR = "#111111"


# -----------------------------------------------------------------------------
# Loading and PR helpers
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

    return y_true, y_pred


def load_flat(root_dir: str, model: str, task: str, seed: int):
    path = os.path.join(root_dir, f"{task}_{model}_seed{seed}_preds.npz")
    return _load_npz(path)


def load_balanced_hierarchical(root_dir: str, model: str, task: str, seed: int):
    path = os.path.join(
        root_dir,
        model,
        f"{task}_balanced",
        f"seed{seed}",
        "preds.npz",
    )
    return _load_npz(path)


def extract_scores(y_pred):
    arr = np.asarray(y_pred)

    if arr.ndim == 1:
        return arr.astype(float)

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        return arr[:, -1].astype(float)

    raise ValueError(f"Unsupported y_pred shape: {arr.shape}")


def compute_pr_curve(y_true, y_pred):
    scores = extract_scores(y_pred)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    # sklearn returns recall in decreasing order; reverse for interpolation/AUC.
    recall = recall[::-1]
    precision = precision[::-1]

    aupr = sklearn_auc(recall, precision)
    prevalence = float(np.mean(y_true))

    return recall, precision, aupr, prevalence


def average_pr(load_fn, root_dir: str, model: str, task: str):
    precisions = []
    auprs = []
    prevalences = []

    for seed in SEEDS:
        loaded = load_fn(root_dir, model, task, seed)
        if loaded is None:
            continue

        y_true, y_pred = loaded
        recall, precision, aupr, prevalence = compute_pr_curve(y_true, y_pred)
        interp_precision = np.interp(MEAN_RECALL, recall, precision)

        precisions.append(interp_precision)
        auprs.append(aupr)
        prevalences.append(prevalence)

    if not precisions:
        return None

    return {
        "mean_precision": np.mean(precisions, axis=0),
        "std_precision": np.std(precisions, axis=0),
        "mean_aupr": float(np.mean(auprs)),
        "std_aupr": float(np.std(auprs)),
        "mean_prevalence": float(np.mean(prevalences)),
        "n": len(precisions),
    }


# -----------------------------------------------------------------------------
# Plot styling helpers
# -----------------------------------------------------------------------------

def bold_tick_labels(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def style_axis(ax, show_xlabel=True, show_ylabel=True):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_aspect("equal", adjustable="box")

    if show_xlabel:
        ax.set_xlabel("Recall", fontsize=18, fontweight="bold", labelpad=8)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("Precision", fontsize=18, fontweight="bold", labelpad=8)
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


def add_panel_label(ax, label: str):
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


def add_curve_with_band(ax, x, y, ystd, color, linestyle, show_bands=True):
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        lw=2.55,
        alpha=0.96,
        solid_capstyle="round",
        zorder=3,
    )

    if show_bands:
        ax.fill_between(
            x,
            np.clip(y - ystd, 0, 1),
            np.clip(y + ystd, 0, 1),
            color=color,
            alpha=0.075,
            linewidth=0,
            zorder=2,
        )


def make_embedding_handles():
    return [
        Line2D(
            [0], [0],
            color=EMBEDDING_COLORS[emb],
            lw=4.0,
            linestyle="-",
            label=EMBEDDING_LABELS[emb],
        )
        for emb in EMBEDDING_ORDER
    ]


def make_condition_handles():
    return [
        Line2D(
            [0], [0],
            color="black",
            lw=3.0,
            linestyle=CONDITION_STYLES[key],
            label=CONDITION_LABELS[key],
        )
        for key in CONDITION_ORDER
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

    # Left block: embedding colors
    legend_ax.text(
        0.04,
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
        bbox_to_anchor=(0.02, 0.75),
        fontsize=16,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.75,
        labelspacing=0.72,
        borderaxespad=0.0,
    )
    legend_ax.add_artist(embedding_legend)
    for text in embedding_legend.get_texts():
        text.set_fontweight("bold")

    # Right block: dataset / training line styles
    legend_ax.text(
        0.68,
        0.82,
        "Dataset / training",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="black",
    )

    condition_legend = legend_ax.legend(
        handles=make_condition_handles(),
        loc="upper left",
        bbox_to_anchor=(0.66, 0.75),
        fontsize=14.5,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.72,
        borderaxespad=0.0,
    )
    legend_ax.add_artist(condition_legend)
    for text in condition_legend.get_texts():
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

    note = (
        "Shaded regions indicate ±1 standard deviation across runs"
        if show_bands
        else "Shaded regions disabled"
    )

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

    legend_ax.text(
        0.08,
        0.08,
        "Curves show seed-averaged precision-recall performance.",
        ha="left",
        va="center",
        fontsize=17.5,
        fontweight="bold",
        color="black",
    )


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------

def plot_figure(unbalanced_dir: str, balanced_dir: str, output_dir: str, show_bands: bool = True):
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
    pr_axes = axes[:-1]

    aupr_rows = []

    for i, ((model_name, model_label), ax) in enumerate(zip(MODELS, pr_axes)):
        missing = []

        for emb in EMBEDDING_ORDER:
            color = EMBEDDING_COLORS[emb]

            allodiverse_task = ALLODIVERSE_TASKS[emb]
            kinsite_task = KINSITE_TASKS[emb]

            curve_specs = [
                (
                    "allodiverse_original",
                    "AlloDiverse original",
                    load_flat,
                    unbalanced_dir,
                    allodiverse_task,
                ),
                (
                    "allodiverse_balanced",
                    "AlloDiverse balanced",
                    load_balanced_hierarchical,
                    balanced_dir,
                    allodiverse_task,
                ),
                (
                    "kinsite",
                    "KinSite",
                    load_flat,
                    unbalanced_dir,
                    kinsite_task,
                ),
            ]

            for condition_key, condition_label, load_fn, root_dir, task in curve_specs:
                result = average_pr(load_fn, root_dir, model_name, task)

                if result is None:
                    missing.append(f"{condition_label} / {task}")
                    continue

                add_curve_with_band(
                    ax,
                    MEAN_RECALL,
                    result["mean_precision"],
                    result["std_precision"],
                    color=color,
                    linestyle=CONDITION_STYLES[condition_key],
                    show_bands=show_bands,
                )

                aupr_rows.append({
                    "model": model_label,
                    "model_id": model_name,
                    "condition": CONDITION_LABELS[condition_key],
                    "task": task,
                    "embedding": EMBEDDING_LABELS[emb],
                    "mean_aupr": result["mean_aupr"],
                    "std_aupr": result["std_aupr"],
                    "mean_prevalence": result["mean_prevalence"],
                    "n_runs": result["n"],
                })

        add_panel_label(ax, PANEL_LABELS[i])
        add_panel_title(ax, model_label)

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
            print(f"[{model_label}] Missing {len(missing)} PR curves:")
            for item in missing:
                print(f"  - {item}")

    draw_legend_panel(legend_ax, show_bands=show_bands)

    fig.suptitle(
        "Precision-recall curves across OneProt encoder architectures",
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

    stem = os.path.join(output_dir, "supplement_all_models_pr_4x2_publication_style")
    pdf_path = f"{stem}.pdf"
    svg_path = f"{stem}.svg"
    png_path = f"{stem}.png"
    tif_path = f"{stem}.tif"
    csv_path = os.path.join(output_dir, "supplement_all_models_pr_aupr_values.csv")

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
                "condition",
                "task",
                "embedding",
                "mean_aupr",
                "std_aupr",
                "mean_prevalence",
                "n_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(aupr_rows)

    print(f"Saved PDF  -> {pdf_path}")
    print(f"Saved SVG  -> {svg_path}")
    print(f"Saved PNG  -> {png_path}")
    print(f"Saved TIFF -> {tif_path}")
    print(f"Saved CSV  -> {csv_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unbalanced_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
        help="Directory with flat-format prediction .npz files.",
    )
    parser.add_argument(
        "--balanced_dir",
        default="/p/scratch/hai_oneprot/curve_plots",
        help="Directory with hierarchical balanced AlloDiverse prediction files.",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/figure3_imbalance_pr",
        help="Output directory.",
    )
    parser.add_argument(
        "--no_bands",
        action="store_true",
        help="Disable shaded ±1 SD bands.",
    )

    args = parser.parse_args()

    print(f"Unbalanced dir: {args.unbalanced_dir}")
    print(f"Balanced dir  : {args.balanced_dir}")
    print(f"Output dir    : {args.output_dir}")

    plot_figure(
        unbalanced_dir=args.unbalanced_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
        show_bands=not args.no_bands,
    )

    print("Done.")


if __name__ == "__main__":
    main()
