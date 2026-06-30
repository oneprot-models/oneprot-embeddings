#!/usr/bin/env python3
"""
plot_figure6_architecture_roc_no_plot_legends.py

Publication-quality architecture-level ROC comparison for a fixed downstream
embedding set across four allosteric-site regimes.

This version is designed to avoid the title/label intersections seen in the
previous v2 figure by:
  - removing all legends from inside the ROC panels;
  - removing any mention of seeds from visible figure text;
  - using a dedicated right-side information column for the embedding-set note,
    shared legend, and shaded-band explanation;
  - separating dataset titles and regime subtitles into separate text objects;
  - coloring the regime subtitle differently from the dataset title;
  - tightening vertical and horizontal spacing between ROC panels;
  - anchoring square ROC axes inward so left and right plot columns sit closer together;
  - keeping the panels large, square, clean, and grid-free.

Outputs:
  figure6_architecture_roc_<embedding_key>_clean.pdf
  figure6_architecture_roc_<embedding_key>_clean.svg
  figure6_architecture_roc_<embedding_key>_clean.png
  figure6_architecture_roc_<embedding_key>_clean.tif
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

MODEL_COLORS = {
    "oneprot_pocket_text_32900": "#4D4D4D",
    "oneprot_struct_token_pocket_text_32900": "#0072B2",
    "oneprot_struct_graph_pocket_text_32900": "#009E73",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "#56B4E9",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "#E69F00",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "#CC79A7",
    "oneprot_md_combined_gpcr_32900": "#D55E00",
}

MODEL_LINESTYLES = {model: "-" for model in MODELS}

# Internal averaging seeds. These are intentionally not written anywhere on the figure.
SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 600)

DATASET_ORDER = ["PPI-Site", "KinSite", "DualSite", "AlloDiverse"]

DATASET_META = {
    "PPI-Site": {
        "panel_label": "A",
        "display_name": "PPI-Site",
        "regime": "Low-separability regime",
        "source": "flat",
    },
    "KinSite": {
        "panel_label": "B",
        "display_name": "KinSite",
        "regime": "Intermediate regime",
        "source": "flat",
    },
    "DualSite": {
        "panel_label": "C",
        "display_name": "Dual-Site",
        "regime": "  Intermediate-synergistic regime",
        "source": "flat",
    },
    "AlloDiverse": {
        "panel_label": "D",
        "display_name": "AlloDiverse",
        "regime": "High-separability regime",
        "source": "balanced",
    },
}

TASK_MAP = {
    "p": {
        "PPI-Site": "ASD_pockets_binary_comp",
        "KinSite": "Kinase_pocket",
        "DualSite": "merged_pocket_binary_comp",
        "AlloDiverse": "ASD_merged_pocket_binary_comp",
    },
    "pt": {
        "PPI-Site": "ASD_pockets_binary_text_comp",
        "KinSite": "Kinase_pocket_text",
        "DualSite": "merged_pocket_binary_text_comp",
        "AlloDiverse": "ASD_merged_pocket_binary_text_comp",
    },
    "ps": {
        "PPI-Site": "ASD_pockets_sequence_binary_comp",
        "KinSite": "Kinase_combined",
        "DualSite": "merged_pocket_sequence_binary_comp",
        "AlloDiverse": "ASD_merged_pocket_sequence_binary_comp",
    },
    "pst": {
        "PPI-Site": "ASD_pockets_sequence_binary_text_comp",
        "KinSite": "Kinase_combined_text",
        "DualSite": "merged_pocket_sequence_binary_text_comp",
        "AlloDiverse": "ASD_merged_pocket_sequence_binary_text_comp",
    },
}

EMBEDDING_LABELS = {
    "p": "Pocket only",
    "pt": "Pocket + Text",
    "ps": "Pocket + Sequence",
    "pst": "Pocket + Sequence + Text",
}

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
    # Slightly closer to the panel than before to avoid title collisions.
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
    # Separate title/subtitle text objects give better vertical control and readability.
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


def make_shared_legend_handles():
    handles = []
    for model in MODELS:
        handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model],
                lw=4.0,
                linestyle=MODEL_LINESTYLES[model],
                label=MODEL_LABELS[model],
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


def draw_right_info_column(info_ax, embedding_key: str):
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axis("off")

    # -------------------------------------------------------------------------
    # Horizontal shift for all right-column content.
    # Increase this value to move the embedding note, legend, separator lines,
    # and shaded-band explanation closer to the ROC panels.
    # Good values to try: 0.06, 0.08, 0.10, 0.12
    # -------------------------------------------------------------------------
    SHIFT_LEFT = 0.50

    x_center = 0.50 - SHIFT_LEFT
    x_left = 0.08 - SHIFT_LEFT
    x_right = 0.92 - SHIFT_LEFT
    x_patch = 0.10 - SHIFT_LEFT
    x_text = 0.30 - SHIFT_LEFT

    # Embedding-set note placed in the right column instead of above the panels.
    info_ax.text(
        x_center,
        0.88,
        "Fixed downstream\nembedding set:",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="black",
        linespacing=1.15,
    )
    info_ax.text(
        x_center,
        0.81,
        EMBEDDING_LABELS[embedding_key],
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=REGIME_COLOR,
        linespacing=1.15,
        wrap=True,
    )

    info_ax.plot([x_left, x_right], [0.72, 0.72], color="black", lw=1.1)

    legend = info_ax.legend(
        handles=make_shared_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(x_left, 0.66),
        fontsize=16,
        frameon=False,
        handlelength=3.2,
        handletextpad=0.9,
        labelspacing=1.0,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    info_ax.plot([x_left, x_right], [0.22, 0.22], color="black", lw=1.1)

    # Shaded-band note without mentioning seeds.
    info_ax.add_patch(
        Rectangle(
            (-0.4, 0.125),
            0.13,
            0.045,
            facecolor="#BFC7D5",
            edgecolor="none",
            alpha=0.65,
            clip_on=False
        )
    )
    info_ax.text(
        x_text,
        0.148,
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

def plot_figure(base_dir: str, balanced_dir: str, output_dir: str, embedding_key: str):
    if embedding_key not in TASK_MAP:
        raise ValueError(f"Unknown embedding_key={embedding_key}. Choose from {list(TASK_MAP)}")

    # Important: because each ROC panel uses ax.set_aspect("equal", adjustable="box"),
    # reducing wspace alone does not fully close the visible gap between columns.
    # Matplotlib keeps each plotting area square and centers it inside its GridSpec cell.
    # We therefore also anchor the left-column axes to the east and the right-column
    # axes to the west after style_axis() is called below.
    fig = plt.figure(figsize=(17.2, 10.8), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.0, 1.0, 0.50],
        height_ratios=[1.0, 1.0],
        wspace=0.01,
        hspace=0.3,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    info_ax = fig.add_subplot(gs[:, 2])

    for idx, (ax, dataset_name) in enumerate(zip(axes, DATASET_ORDER)):
        task = TASK_MAP[embedding_key][dataset_name]
        meta = DATASET_META[dataset_name]
        missing = []

        for model in MODELS:
            if meta["source"] == "balanced":
                result = average_roc(load_balanced_hierarchical, balanced_dir, model, task)
            else:
                result = average_roc(load_flat, base_dir, model, task)

            if result is None:
                missing.append(MODEL_LABELS[model])
                continue

            ax.plot(
                MEAN_FPR,
                result["mean_tpr"],
                color=MODEL_COLORS[model],
                lw=3.2,
                linestyle=MODEL_LINESTYLES[model],
                solid_capstyle="round",
                zorder=3,
            )
            ax.fill_between(
                MEAN_FPR,
                np.clip(result["mean_tpr"] - result["std_tpr"], 0, 1),
                np.clip(result["mean_tpr"] + result["std_tpr"], 0, 1),
                color=MODEL_COLORS[model],
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

        add_panel_label(ax, meta["panel_label"])
        add_panel_title(ax, meta["display_name"], meta["regime"])

        style_axis(
            ax,
            show_xlabel=idx in [2, 3],
            show_ylabel=idx in [0, 2],
        )

        # Pull square ROC panels slightly inward, but leave a small visible gap.
        # Because ax.set_aspect("equal", adjustable="box") keeps each ROC panel square,
        # wspace alone has limited control over the visible distance between panels.
        # Fractional anchors are less aggressive than "E" / "W" and prevent the
        # left and right panels from touching.
        if idx in [0, 2]:      # panels A and C: left column
            ax.set_anchor((0.8, 0.50))
        elif idx in [1, 3]:    # panels B and D: right column
            ax.set_anchor((0.18, 0.50))

        if missing:
            print(f"[{dataset_name}] Missing models:")
            for model_label in missing:
                print(f"  - {model_label}")

    draw_right_info_column(info_ax, embedding_key)

    fig.suptitle(
        "Architecture-dependent ROC curves across allosteric-site regimes",
        fontsize=25,
        fontweight="bold",
        y=0.985,
    )

    # Manual margins preserve large panels. The horizontal closeness is now controlled
    # mainly by the inward anchoring above, not only by wspace.
    fig.subplots_adjust(
        left=0.065,
        right=0.975,
        bottom=0.085,
        top=0.875,
        wspace=0.01,
        hspace=0.3,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, f"figure6_architecture_roc_{embedding_key}_clean")

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
        default="<REPO_ROOT>/figure6_model_comparison",
        help="Output directory.",
    )
    parser.add_argument(
        "--embedding_key",
        default="pst",
        choices=["p", "pt", "ps", "pst"],
        help="Fixed downstream embedding set.",
    )

    args = parser.parse_args()
    print(f"Base dir      : {args.base_dir}")
    print(f"Balanced dir  : {args.balanced_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Embedding set : {EMBEDDING_LABELS[args.embedding_key]}")

    plot_figure(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
        embedding_key=args.embedding_key,
    )
    print("Done.")


if __name__ == "__main__":
    main()
