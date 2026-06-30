"""
plot_auc_violin_by_dataset.py

Create violin plots of ROC-AUC distributions across datasets.

Each violin pools ROC-AUC values across:
    - model variants
    - random seeds

Default embedding set:
    pocket+sequence+text ("pst")

Usage:
    python plot_auc_violin_by_dataset.py \
        --base_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots \
        --balanced_dir /p/scratch/hai_oneprot/curve_plots \
        --output_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/separability_figures \
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
    "oneprot_pocket_text_32900": "pocket_text",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "full_allatom",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "md_no_graph",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "md_no_token",
    "oneprot_struct_graph_pocket_text_32900": "struct_graph",
    "oneprot_md_combined_gpcr_32900": "md_combined",
    "oneprot_struct_token_pocket_text_32900": "struct_token",
}

SEEDS = [11, 12, 13, 14, 15]

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

DATASET_ORDER = [
    "PL8",
    "Kinase",
    "PL8 + Kinase",
    "ASD + PL8 + Kinase",
]


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
    roc_auc = sklearn_auc(fpr, tpr)

    return roc_auc


def load_flat(base_dir: str, model: str, task: str, seed: int):
    path = os.path.join(
        base_dir,
        f"{task}_{model}_seed{seed}_preds.npz",
    )
    return _load_npz(path)


def load_balanced_hierarchical(
    balanced_dir: str,
    model: str,
    task: str,
    seed: int,
):
    path = os.path.join(
        balanced_dir,
        model,
        f"{task}_balanced",
        f"seed{seed}",
        "preds.npz",
    )
    return _load_npz(path)


# ── Data collection ───────────────────────────────────────────────────────────

def collect_auc_values(
    base_dir: str,
    balanced_dir: str,
    embedding_key: str,
):
    rows = []
    missing = {}

    for dataset_name in DATASET_ORDER:
        task = TASK_MAP[embedding_key][dataset_name]
        missing[dataset_name] = []

        for model in MODELS:
            model_label = MODEL_LABELS[model]

            for seed in SEEDS:
                if dataset_name == "ASD + PL8 + Kinase":
                    roc_auc = load_balanced_hierarchical(
                        balanced_dir,
                        model,
                        task,
                        seed,
                    )
                else:
                    roc_auc = load_flat(
                        base_dir,
                        model,
                        task,
                        seed,
                    )

                if roc_auc is None:
                    missing[dataset_name].append((model_label, seed))
                    continue

                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_label,
                        "seed": seed,
                        "auc": float(roc_auc),
                    }
                )

    return rows, missing


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_auc_violins(
    base_dir: str,
    balanced_dir: str,
    output_dir: str,
    embedding_key: str,
):
    if embedding_key not in TASK_MAP:
        raise ValueError(
            f"Unknown embedding_key={embedding_key}. "
            f"Choose from {list(TASK_MAP.keys())}"
        )

    rows, missing = collect_auc_values(
        base_dir=base_dir,
        balanced_dir=balanced_dir,
        embedding_key=embedding_key,
    )

    data_by_dataset = []
    summary_lines = []

    for dataset_name in DATASET_ORDER:
        aucs = np.array(
            [r["auc"] for r in rows if r["dataset"] == dataset_name],
            dtype=float,
        )

        if len(aucs) == 0:
            raise RuntimeError(f"No AUC values found for dataset: {dataset_name}")

        data_by_dataset.append(aucs)

        summary_lines.append(
            f"{dataset_name}: "
            f"n={len(aucs)}, "
            f"mean={np.mean(aucs):.3f}, "
            f"median={np.median(aucs):.3f}, "
            f"std={np.std(aucs):.3f}, "
            f"min={np.min(aucs):.3f}, "
            f"max={np.max(aucs):.3f}"
        )

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    positions = np.arange(1, len(DATASET_ORDER) + 1)

    violin_parts = ax.violinplot(
        data_by_dataset,
        positions=positions,
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.75,
    )

    for body in violin_parts["bodies"]:
        body.set_alpha(0.55)

    rng = np.random.default_rng(123)

    for i, aucs in enumerate(data_by_dataset, start=1):
        x_jitter = rng.normal(i, 0.045, size=len(aucs))

        ax.scatter(
            x_jitter,
            aucs,
            s=22,
            alpha=0.55,
            edgecolor="none",
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.75,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(DATASET_ORDER, rotation=20, ha="right")

    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_ylim(0.45, 1.02)

    ax.set_title(
        "Dataset-dependent separability regimes of allosteric discrimination\n"
        f"Embedding set: {EMBEDDING_LABELS[embedding_key]}",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(1, 0.465, "low", ha="center", fontsize=9)
    ax.text(2.5, 0.465, "intermediate", ha="center", fontsize=9)
    ax.text(4, 0.465, "high", ha="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)

    out_png = os.path.join(
        output_dir,
        f"auc_violin_by_dataset_{embedding_key}_balancedASD.png",
    )
    out_pdf = os.path.join(
        output_dir,
        f"auc_violin_by_dataset_{embedding_key}_balancedASD.pdf",
    )
    out_txt = os.path.join(
        output_dir,
        f"auc_violin_by_dataset_{embedding_key}_summary.txt",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    with open(out_txt, "w") as f:
        f.write(
            f"Embedding key: {embedding_key} "
            f"({EMBEDDING_LABELS[embedding_key]})\n\n"
        )
        f.write("\n".join(summary_lines))
        f.write("\n\nMissing files / skipped entries:\n")

        for dataset_name, missing_entries in missing.items():
            f.write(f"\n{dataset_name}: {len(missing_entries)} missing\n")
            for model_label, seed in missing_entries:
                f.write(f"  {model_label}, seed {seed}\n")

    print("\nSummary:")
    for line in summary_lines:
        print("  " + line)

    for dataset_name, missing_entries in missing.items():
        if missing_entries:
            print(f"[{dataset_name}] Missing/skipped entries: {len(missing_entries)}")

    print(f"\nSaved PNG → {out_png}")
    print(f"Saved PDF → {out_pdf}")
    print(f"Saved summary → {out_txt}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
        help="Directory with flat-format .npz prediction files.",
    )
    parser.add_argument(
        "--balanced_dir",
        default="/p/scratch/hai_oneprot/curve_plots",
        help="Directory with hierarchical balanced ASD prediction files.",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/separability_figures",
        help="Output directory for violin plots.",
    )
    parser.add_argument(
        "--embedding_key",
        default="pst",
        choices=["p", "pt", "ps", "pst"],
        help="Embedding set to compare across datasets.",
    )

    args = parser.parse_args()

    print(f"Base dir      : {args.base_dir}")
    print(f"Balanced dir  : {args.balanced_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(
        f"Embedding key : {args.embedding_key} "
        f"({EMBEDDING_LABELS[args.embedding_key]})"
    )

    plot_auc_violins(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
        embedding_key=args.embedding_key,
    )

    print("Done.")


if __name__ == "__main__":
    main()