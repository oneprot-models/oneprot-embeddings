#!/usr/bin/env python3
"""
plot_figure2_representative_roc_publication_style.py

Publication-quality representative ROC figure for one fixed OneProt architecture
across four allosteric-site regimes.

This is styled to match the Figure 6 architecture-comparison style:
  - clean 2x2 ROC panels;
  - no legends inside individual panels;
  - shared right-side information column;
  - separate dataset titles and regime subtitles;
  - bold, large axis labels and tick labels;
  - no visible grid;
  - square ROC panels;
  - high-resolution outputs: PDF, SVG, PNG, and TIFF.

Outputs:
  figure2_representative_roc_publication_style.pdf
  figure2_representative_roc_publication_style.svg
  figure2_representative_roc_publication_style.png
  figure2_representative_roc_publication_style.tif
"""

import os
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

    "lines.linewidth": 3.0,
    "legend.fontsize": 13,
    "legend.frameon": False,

    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL = "oneprot_md_combined_gpcr_no_struct_graph_32900"
MODEL_LABEL = "MD+ST+Pocket+Text"

# Internal averaging seeds. These are intentionally not written anywhere on the figure.
SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 600)

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket only",
    "pt": "Pocket+Text",
    "ps": "Pocket+Sequence",
    "pst": "Pocket+Sequence+Text",
}

# Same embedding colors as the original figure/script.
EMBEDDING_COLORS = {
    "p": "#1f77b4",
    "pt": "#ff7f0e",
    "ps": "#2ca02c",
    "pst": "#d62728",
}

DATASETS = [
    {
        "panel_label": "A",
        "panel_title": "PPI-Site",
        "regime": "Low-separability regime",
        "source": "flat",
        "tasks": {
            "p": "ASD_pockets_binary_comp",
            "pt": "ASD_pockets_binary_text_comp",
            "ps": "ASD_pockets_sequence_binary_comp",
            "pst": "ASD_pockets_sequence_binary_text_comp",
        },
    },
    {
        "panel_label": "B",
        "panel_title": "KinSite",
        "regime": "Intermediate regime",
        "source": "flat",
        "tasks": {
            "p": "Kinase_pocket",
            "pt": "Kinase_pocket_text",
            "ps": "Kinase_combined",
            "pst": "Kinase_combined_text",
        },
    },
    {
        "panel_label": "C",
        "panel_title": "Dual-Site",
        "regime": "   Intermediate-synergistic regime",
        "source": "flat",
        "tasks": {
            "p": "merged_pocket_binary_comp",
            "pt": "merged_pocket_binary_text_comp",
            "ps": "merged_pocket_sequence_binary_comp",
            "pst": "merged_pocket_sequence_binary_text_comp",
        },
    },
    {
        "panel_label": "D",
        "panel_title": "AlloDiverse",
        "regime": "High-separability regime",
        "source": "balanced",
        "tasks": {
            "p": "ASD_merged_pocket_binary_comp",
            "pt": "ASD_merged_pocket_binary_text_comp",
            "ps": "ASD_merged_pocket_sequence_binary_comp",
            "pst": "ASD_merged_pocket_sequence_binary_text_comp",
        },
    },
]

REGIME_COLOR = "#0B3D91"
TITLE_COLOR = "#111111"


# -----------------------------------------------------------------------------
# Loading helpers
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


def load_flat(base_dir: str, model: str, task: str, seed: int):
    path = os.path.join(base_dir, f"{task}_{model}_seed{seed}_preds.npz")
    return _load_npz(path)


def load_balanced_hierarchical(balanced_dir: str, model: str, task: str, seed: int):
    path = os.path.join(
        balanced_dir,
        model,
        f"{task}_balanced",
        f"seed{seed}",
        "preds.npz",
    )
    return _load_npz(path)


def average_roc(load_fn, root_dir: str, model: str, task: str):
    tprs, aucs = [], []

    for seed in SEEDS:
        result = load_fn(root_dir, model, task, seed)
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
# Plotting helpers
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


def add_panel_title(ax, dataset_title: str, regime_subtitle: str):
    ax.text(
        0.5,
        1.085,
        dataset_title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=19,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    ax.text(
        0.5,
        1.025,
        regime_subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15.5,
        fontweight="bold",
        fontstyle="italic",
        color=REGIME_COLOR,
    )


def make_embedding_legend_handles():
    handles = []
    for emb_key in EMBEDDING_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                color=EMBEDDING_COLORS[emb_key],
                lw=4.0,
                linestyle="-",
                label=EMBEDDING_LABELS[emb_key],
            )
        )

    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.0,
            linestyle="--",
            label="Random classifier",
        )
    )
    return handles


def draw_right_info_column(info_ax):
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axis("off")

    # Shift all right-column content closer to ROC panels.
    # Increase to move farther left; decrease to move farther right.
    SHIFT_LEFT = 0.60

    x_center = 0.50 - SHIFT_LEFT
    x_left = 0.08 - SHIFT_LEFT
    x_right = 0.92 - SHIFT_LEFT
    x_patch = 0.10 - SHIFT_LEFT
    x_text = 0.30 - SHIFT_LEFT

    info_ax.text(
        x_center,
        0.88,
        "Representative\narchitecture:",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="black",
        linespacing=1.15,
    )
    info_ax.text(
        x_center,
        0.80,
        MODEL_LABEL,
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=REGIME_COLOR,
        linespacing=1.15,
        wrap=True,
    )

    info_ax.plot([x_left, x_right], [0.70, 0.70], color="black", lw=1.1)

    legend = info_ax.legend(
        handles=make_embedding_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(x_left, 0.64),
        fontsize=16,
        frameon=False,
        handlelength=3.2,
        handletextpad=0.9,
        labelspacing=1.0,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    info_ax.plot([x_left, x_right], [0.31, 0.31], color="black", lw=1.1)

    info_ax.add_patch(
        Rectangle(
            (x_patch, 0.205),
            0.13,
            0.045,
            facecolor="#BFC7D5",
            edgecolor="none",
            alpha=0.65,
            clip_on=False,
            zorder=20,
        )
    )
    info_ax.text(
        x_text,
        0.228,
        "Shaded regions\nindicate ±1 standard\ndeviation across runs",
        ha="left",
        va="center",
        fontsize=16.5,
        color="black",
        linespacing=1.2,
    )


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------

def plot_figure(base_dir: str, balanced_dir: str, output_dir: str):
    fig = plt.figure(figsize=(17.2, 10.8), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.0, 1.0, 0.50],
        height_ratios=[1.0, 1.0],
        wspace=0.01,
        hspace=0.30,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    info_ax = fig.add_subplot(gs[:, 2])

    for idx, (ax, dataset_cfg) in enumerate(zip(axes, DATASETS)):
        missing = []

        for emb_key in EMBEDDING_ORDER:
            task = dataset_cfg["tasks"][emb_key]
            color = EMBEDDING_COLORS[emb_key]

            if dataset_cfg["source"] == "flat":
                result = average_roc(load_flat, base_dir, MODEL, task)
            else:
                result = average_roc(load_balanced_hierarchical, balanced_dir, MODEL, task)

            if result is None:
                missing.append(task)
                continue

            ax.plot(
                MEAN_FPR,
                result["mean_tpr"],
                color=color,
                lw=3.2,
                linestyle="-",
                solid_capstyle="round",
                zorder=3,
            )

            ax.fill_between(
                MEAN_FPR,
                np.clip(result["mean_tpr"] - result["std_tpr"], 0, 1),
                np.clip(result["mean_tpr"] + result["std_tpr"], 0, 1),
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=2,
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            lw=2.0,
            alpha=0.75,
            zorder=1,
        )

        add_panel_label(ax, dataset_cfg["panel_label"])
        add_panel_title(ax, dataset_cfg["panel_title"], dataset_cfg["regime"])

        style_axis(
            ax,
            show_xlabel=idx in [2, 3],
            show_ylabel=idx in [0, 2],
        )

        # Same inward anchoring concept as the current Figure 6 script.
        # Tune these two x values if the panel gap is not exactly as desired.
        if idx in [0, 2]:
            ax.set_anchor((0.80, 0.50))
        elif idx in [1, 3]:
            ax.set_anchor((0.18, 0.50))

        if missing:
            print(f"[{dataset_cfg['panel_title']}] Missing tasks:")
            for task in missing:
                print(f"  - {task}")

    draw_right_info_column(info_ax)

    fig.suptitle(
        "Representative ROC curves across allosteric-site regimes",
        fontsize=25,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.975,
        bottom=0.085,
        top=0.875,
        wspace=0.01,
        hspace=0.30,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, "figure2_representative_roc_publication_style")

    out_pdf = f"{stem}.pdf"
    out_svg = f"{stem}.svg"
    out_png = f"{stem}.png"
    out_tif = f"{stem}.tif"

    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_tif, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"Saved PDF  -> {out_pdf}")
    print(f"Saved SVG  -> {out_svg}")
    print(f"Saved PNG  -> {out_png}")
    print(f"Saved TIFF -> {out_tif}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="<REPO_ROOT>/curve_plots",
        help="Directory with flat-format prediction .npz files.",
    )
    parser.add_argument(
        "--balanced_dir",
        default="<CURVE_PLOTS_ROOT>",
        help="Directory with hierarchical balanced AlloDiverse prediction files.",
    )
    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/figure2_representative",
        help="Output directory.",
    )

    args = parser.parse_args()
    print(f"Base dir     : {args.base_dir}")
    print(f"Balanced dir : {args.balanced_dir}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Architecture : {MODEL_LABEL}")

    plot_figure(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
