"""
plot_figure6_model_comparison_roc_clean.py

Architecture-level ROC comparison for a fixed embedding set across dataset regimes.

Design:
    - fixed downstream embedding set: default = Pocket+Sequence+Text ("pst")
    - 4 subplots = dataset regimes:
        A) PPI-Site
        B) KinSite
        C) DualSite
        D) AlloDiverse
    - 7 curves per subplot = OneProt encoder architecture variants

Outputs:
    - figure6_architecture_roc_<embedding_key>.png
    - figure6_architecture_roc_<embedding_key>.pdf
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc as sklearn_auc


# ── Configuration ─────────────────────────────────────────────────────────────

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
    "oneprot_pocket_text_32900": "#7f7f7f",
    "oneprot_struct_token_pocket_text_32900": "#1f77b4",
    "oneprot_struct_graph_pocket_text_32900": "#2ca02c",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "#17becf",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "#ff7f0e",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "#9467bd",
    "oneprot_md_combined_gpcr_32900": "#d62728",
}

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 300)

DATASET_ORDER = ["PPI-Site", "KinSite", "DualSite", "AlloDiverse"]

DATASET_META = {
    "PPI-Site": {
        "panel_label": "A",
        "regime": "Low separability",
        "source": "flat",
    },
    "KinSite": {
        "panel_label": "B",
        "regime": "Intermediate",
        "source": "flat",
    },
    "DualSite": {
        "panel_label": "C",
        "regime": "Intermediate-synergistic",
        "source": "flat",
    },
    "AlloDiverse": {
        "panel_label": "D",
        "regime": "High separability",
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
    "pt": "Pocket+Text",
    "ps": "Pocket+Sequence",
    "pst": "Pocket+Sequence+Text",
}


# ── Loading helpers ───────────────────────────────────────────────────────────

def _load_npz(path: str):
    if not os.path.exists(path):
        return None

    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr, sklearn_auc(fpr, tpr)


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


# ── Plotting ──────────────────────────────────────────────────────────────────

def style_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel("False positive rate", fontsize=14)
    ax.set_ylabel("True positive rate", fontsize=14)

    ax.tick_params(axis="both", labelsize=12, width=1.1, length=4)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    ax.grid(True, linestyle=":", linewidth=0.75, alpha=0.45)


def plot_figure(base_dir: str, balanced_dir: str, output_dir: str, embedding_key: str):
    if embedding_key not in TASK_MAP:
        raise ValueError(
            f"Unknown embedding_key={embedding_key}. "
            f"Choose from {list(TASK_MAP.keys())}"
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 10.8),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, dataset_name in zip(axes, DATASET_ORDER):
        task = TASK_MAP[embedding_key][dataset_name]
        meta = DATASET_META[dataset_name]
        missing = []

        for model in MODELS:
            color = MODEL_COLORS[model]
            model_label = MODEL_LABELS[model]

            if meta["source"] == "balanced":
                result = average_roc(
                    load_balanced_hierarchical,
                    balanced_dir,
                    model,
                    task,
                )
            else:
                result = average_roc(
                    load_flat,
                    base_dir,
                    model,
                    task,
                )

            if result is None:
                missing.append(model_label)
                continue

            label = (
                f"{model_label} "
                f"(AUC={result['mean_auc']:.2f}±{result['std_auc']:.2f})"
            )

            ax.plot(
                MEAN_FPR,
                result["mean_tpr"],
                color=color,
                lw=2.15,
                label=label,
            )

            ax.fill_between(
                MEAN_FPR,
                np.clip(result["mean_tpr"] - result["std_tpr"], 0, 1),
                np.clip(result["mean_tpr"] + result["std_tpr"], 0, 1),
                color=color,
                alpha=0.085,
                linewidth=0,
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            lw=1.1,
            alpha=0.65,
        )

        ax.set_title(
            f"{meta['panel_label']}. {dataset_name} ({meta['regime']})",
            fontsize=17,
            fontweight="bold",
            pad=16,
        )

        style_axis(ax)

        ax.legend(
            fontsize=9.4,
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            borderpad=0.55,
            handlelength=2.0,
            labelspacing=0.38,
        )

        if missing:
            print(f"[{dataset_name}] Missing models:")
            for model_label in missing:
                print(f"  - {model_label}")

    fig.suptitle(
        "Architecture-dependent ROC curves across allosteric-site regimes\n"
        f"Fixed downstream embedding set: {EMBEDDING_LABELS[embedding_key]}",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )

    # Explicit spacing prevents title/panel overlap without reducing font size.
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.155,
        top=0.845,
        wspace=0.18,
        hspace=0.34,
    )

    shared_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            lw=3,
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]

    shared_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.5,
            linestyle="--",
            label="Random (AUC=0.50)",
        )
    )

    fig.legend(
        handles=shared_handles,
        loc="lower center",
        ncol=4,
        fontsize=11.2,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.035),
        handlelength=2.7,
        columnspacing=1.5,
        borderpad=0.85,
    )

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(
        output_dir,
        f"figure6_architecture_roc_{embedding_key}.png",
    )
    out_pdf = os.path.join(
        output_dir,
        f"figure6_architecture_roc_{embedding_key}.pdf",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG -> {out_png}")
    print(f"Saved PDF -> {out_pdf}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
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
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/figure6_model_comparison",
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