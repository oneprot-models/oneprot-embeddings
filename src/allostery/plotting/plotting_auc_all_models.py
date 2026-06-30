"""
plot_figure2_model_comparison_roc.py

Figure 2: model comparison for a fixed embedding set across datasets.

Design:
    - fixed embedding set: default = pocket+sequence+text ("pst")
    - 4 subplots = dataset regimes
        1) PL8
        2) Kinase
        3) PL8 + Kinase
        4) ASD + PL8 + Kinase   (BALANCED ASD run)
    - 7 curves per subplot = model variants

Data sources:
    - PL8, merged, Kinase -> flat naming
          <base_dir>/{task}_{model}_seed{seed}_preds.npz
    - BALANCED ASD_merged -> hierarchical naming
          <balanced_dir>/<model>/{task}_balanced/seed{seed}/preds.npz

Usage:
    python plot_figure2_model_comparison_roc.py \
        --base_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots \
        --balanced_dir /p/scratch/hai_oneprot/curve_plots \
        --output_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/figure2_model_comparison \
        --embedding_key pst
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc


# ── Configuration ─────────────────────────────────────────────────────────────

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

MODEL_COLORS = {
    "oneprot_pocket_text_32900": "#1f77b4",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "#ff7f0e",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "#2ca02c",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "#d62728",
    "oneprot_struct_graph_pocket_text_32900": "#9467bd",
    "oneprot_md_combined_gpcr_32900": "#8c564b",
    "oneprot_struct_token_pocket_text_32900": "#e377c2",
}

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 200)

# Embedding-set task mapping for each dataset
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

EMBEDDING_LABELS = {
    "p": "pocket",
    "pt": "pocket+text",
    "ps": "pocket+sequence",
    "pst": "pocket+sequence+text",
}

DATASET_ORDER = ["PL8", "Kinase", "PL8 + Kinase", "ASD + PL8 + Kinase"]


# ── Loading helpers ───────────────────────────────────────────────────────────

def load_flat(base_dir: str, model: str, task: str, seed: int):
    path = os.path.join(base_dir, f"{task}_{model}_seed{seed}_preds.npz")
    return _load_npz(path)

def load_balanced_hierarchical(balanced_dir: str, model: str, task: str, seed: int):
    path = os.path.join(balanced_dir, model, f"{task}_balanced", f"seed{seed}", "preds.npz")
    return _load_npz(path)

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
    std_tpr = np.std(tprs, axis=0)

    return {
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "n": len(tprs),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_figure(base_dir: str, balanced_dir: str, output_dir: str, embedding_key: str):
    if embedding_key not in TASK_MAP:
        raise ValueError(f"Unknown embedding_key={embedding_key}. Choose from {list(TASK_MAP.keys())}")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.5, 9.5),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for ax, dataset_name in zip(axes, DATASET_ORDER):
        task = TASK_MAP[embedding_key][dataset_name]
        missing = []

        for model in MODELS:
            color = MODEL_COLORS[model]
            model_label = MODEL_LABELS[model]

            if dataset_name == "ASD + PL8 + Kinase":
                result = average_roc(load_balanced_hierarchical, balanced_dir, model, task)
            else:
                result = average_roc(load_flat, base_dir, model, task)

            if result is None:
                missing.append(model_label)
                continue

            mean_tpr = result["mean_tpr"]
            std_tpr = result["std_tpr"]
            mean_auc = result["mean_auc"]
            std_auc = result["std_auc"]
            n = result["n"]

            n_prefix = f"(n={n}) " if n < len(SEEDS) else ""
            label = f"{n_prefix}{model_label}  {mean_auc:.2f}±{std_auc:.2f}"

            ax.plot(
                MEAN_FPR,
                mean_tpr,
                color=color,
                lw=1.8,
                label=label,
            )
            ax.fill_between(
                MEAN_FPR,
                np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color=color,
                alpha=0.10,
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate", fontsize=10)
        ax.set_ylabel("True positive rate", fontsize=10)
        ax.set_title(dataset_name, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=7.2, loc="lower right", framealpha=0.9)

        if missing:
            print(f"[{dataset_name}] Missing models: {missing}")

    fig.suptitle(
        f"Model comparison across datasets for fixed embeddings: {EMBEDDING_LABELS[embedding_key]}\n"
        f"(balanced ASD split; seeds {SEEDS[0]}–{SEEDS[-1]})",
        fontsize=14,
        fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"figure2_model_comparison_{embedding_key}_balancedASD_roc.png"
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
        help="Directory with flat-format ROC .npz files.",
    )
    parser.add_argument(
        "--balanced_dir",
        default="/p/scratch/hai_oneprot/curve_plots",
        help="Directory with hierarchical balanced ASD results.",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/figure2_model_comparison",
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--embedding_key",
        default="pst",
        choices=["p", "pt", "ps", "pst"],
        help="Fixed embedding set to compare across models.",
    )
    args = parser.parse_args()

    print(f"Base dir      : {args.base_dir}")
    print(f"Balanced dir  : {args.balanced_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Embedding key : {args.embedding_key} ({EMBEDDING_LABELS[args.embedding_key]})")
    plot_figure(args.base_dir, args.balanced_dir, args.output_dir, args.embedding_key)
    print("Done.")

if __name__ == "__main__":
    main()