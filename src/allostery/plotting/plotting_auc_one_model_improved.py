"""
plot_figure2_representative_roc_clean.py

Creates a clean 2x2 ROC figure with:
- separate title area
- separate shared legend area
- no overlap between figure title and panel titles
- updated dataset names: PPI-Site, KinSite, Dual-Site, AlloDiverse
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc as sklearn_auc


MODEL = "oneprot_md_combined_gpcr_no_struct_graph_32900"
MODEL_LABEL = "MD+ST+Pocket+Text"

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 300)

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket only",
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

DATASETS = [
    {
        "panel_label": "A",
        "panel_title": "PPI-Site",
        "regime": "Low separability",
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
        "regime": "Intermediate",
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
        "regime": "Intermediate-synergistic",
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
        "regime": "High separability",
        "source": "balanced",
        "tasks": {
            "p": "ASD_merged_pocket_binary_comp",
            "pt": "ASD_merged_pocket_binary_text_comp",
            "ps": "ASD_merged_pocket_sequence_binary_comp",
            "pst": "ASD_merged_pocket_sequence_binary_text_comp",
        },
    },
]


def _load_npz(path):
    if not os.path.exists(path):
        return None

    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr, sklearn_auc(fpr, tpr)


def load_flat(base_dir, model, task, seed):
    path = os.path.join(base_dir, f"{task}_{model}_seed{seed}_preds.npz")
    return _load_npz(path)


def load_balanced_hierarchical(balanced_dir, model, task, seed):
    path = os.path.join(
        balanced_dir,
        model,
        f"{task}_balanced",
        f"seed{seed}",
        "preds.npz",
    )
    return _load_npz(path)


def average_roc(load_fn, root_dir, model, task):
    tprs, aucs = [], []

    for seed in SEEDS:
        result = load_fn(root_dir, model, task, seed)
        if result is None:
            continue

        fpr, tpr, roc_auc = result
        interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    if not tprs:
        return None

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    return {
        "mean_tpr": mean_tpr,
        "std_tpr": np.std(tprs, axis=0),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "n": len(tprs),
    }


def style_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel("False positive rate", fontsize=14)
    ax.set_ylabel("True positive rate", fontsize=14)

    ax.tick_params(axis="both", labelsize=12, width=1.1, length=4)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    ax.grid(True, linestyle=":", linewidth=0.75, alpha=0.45)


def plot_figure(base_dir, balanced_dir, output_dir):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 10.8),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, dataset_cfg in zip(axes, DATASETS):
        missing = []

        for emb_key in EMBEDDING_ORDER:
            task = dataset_cfg["tasks"][emb_key]
            color = EMBEDDING_COLORS[emb_key]

            if dataset_cfg["source"] == "flat":
                result = average_roc(load_flat, base_dir, MODEL, task)
            else:
                result = average_roc(
                    load_balanced_hierarchical,
                    balanced_dir,
                    MODEL,
                    task,
                )

            if result is None:
                missing.append(task)
                continue

            label = (
                f"{EMBEDDING_LABELS[emb_key]} "
                f"(AUC={result['mean_auc']:.2f}±{result['std_auc']:.2f})"
            )

            ax.plot(
                MEAN_FPR,
                result["mean_tpr"],
                color=color,
                lw=2.5,
                label=label,
            )

            ax.fill_between(
                MEAN_FPR,
                np.clip(result["mean_tpr"] - result["std_tpr"], 0, 1),
                np.clip(result["mean_tpr"] + result["std_tpr"], 0, 1),
                color=color,
                alpha=0.13,
                linewidth=0,
            )

        ax.plot([0, 1], [0, 1], "--", color="black", lw=1.1, alpha=0.65)

        ax.set_title(
            f"{dataset_cfg['panel_label']}. {dataset_cfg['panel_title']} "
            f"({dataset_cfg['regime']})",
            fontsize=17,
            fontweight="bold",
            pad=16,
        )

        style_axis(ax)

        ax.legend(
            fontsize=10,
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            borderpad=0.6,
            handlelength=2.2,
            labelspacing=0.45,
        )

        if missing:
            print(f"[{dataset_cfg['panel_title']}] Missing tasks:")
            for task in missing:
                print(f"  - {task}")

    fig.suptitle(
        "ROC curves across allosteric-site separability regimes\n"
        f"Representative OneProt architecture: {MODEL_LABEL}",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )

    # This is the key fix: reserve explicit space for the title and bottom legend.
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.145,
        top=0.845,
        wspace=0.18,
        hspace=0.34,
    )

    shared_handles = [
        Line2D([0], [0], color=EMBEDDING_COLORS[k], lw=3, label=EMBEDDING_LABELS[k])
        for k in EMBEDDING_ORDER
    ]
    shared_handles.append(
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Random (AUC=0.50)")
    )

    fig.legend(
        handles=shared_handles,
        loc="lower center",
        ncol=5,
        fontsize=13,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.035),
        handlelength=3.0,
        columnspacing=2.0,
        borderpad=0.9,
    )

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(output_dir, "figure2_representative_roc_clean.png")
    out_pdf = os.path.join(output_dir, "figure2_representative_roc_clean.pdf")

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
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/figure2_representative",
    )

    args = parser.parse_args()

    plot_figure(args.base_dir, args.balanced_dir, args.output_dir)


if __name__ == "__main__":
    main()