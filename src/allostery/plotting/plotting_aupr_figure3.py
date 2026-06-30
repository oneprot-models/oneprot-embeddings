"""
plot_figure3_representative_pr.py

Figure 3: representative Precision-Recall comparison for a single model across datasets.

Design:
    - fixed model: md_no_graph
    - 4 subplots = dataset regimes
        1) PL8
        2) Kinase
        3) PL8 + Kinase
        4) ASD + PL8 + Kinase   (BALANCED ASD run)
    - 4 curves per subplot = embedding sets
        pocket
        pocket+text
        pocket+sequence
        pocket+sequence+text

Data sources:
    - PL8, merged, Kinase -> flat naming from evaluate.py
          <base_dir>/{task}_{model}_seed{seed}_preds.npz
    - BALANCED ASD_merged -> hierarchical naming
          <balanced_dir>/<model>/{task}_balanced/seed{seed}/preds.npz

Notes:
    - PR curves are averaged across seeds by interpolating precision
      on a common recall grid.
    - A horizontal dotted line marks the average positive prevalence
      across seeds for that task/dataset, which serves as the random baseline.

Usage:
    python plot_figure3_representative_pr.py \
        --base_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots \
        --balanced_dir /p/scratch/hai_oneprot/curve_plots \
        --output_dir /p/project1/hai_oneprot/bazarova1/oneprot-panda/figure3_representative_pr
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "oneprot_md_combined_gpcr_no_struct_graph_32900"
MODEL_LABEL = "md_no_graph"

SEEDS = [11, 12, 13, 14, 15]
MEAN_RECALL = np.linspace(0, 1, 200)

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]
EMBEDDING_LABELS = {
    "p": "pocket",
    "pt": "pocket+text",
    "ps": "pocket+sequence",
    "pst": "pocket+sequence+text",
}

EMBEDDING_COLORS = {
    "p":   "#1f77b4",  # blue
    "pt":  "#ff7f0e",  # orange
    "ps":  "#2ca02c",  # green
    "pst": "#d62728",  # red
}

DATASETS = [
    {
        "panel_title": "PL8",
        "source": "flat",
        "tasks": {
            "p":   "ASD_pockets_binary_comp",
            "pt":  "ASD_pockets_binary_text_comp",
            "ps":  "ASD_pockets_sequence_binary_comp",
            "pst": "ASD_pockets_sequence_binary_text_comp",
        },
    },
    {
        "panel_title": "Kinase",
        "source": "flat",
        "tasks": {
            "p":   "Kinase_pocket",
            "pt":  "Kinase_pocket_text",
            "ps":  "Kinase_combined",
            "pst": "Kinase_combined_text",
        },
    },
    {
        "panel_title": "PL8 + Kinase",
        "source": "flat",
        "tasks": {
            "p":   "merged_pocket_binary_comp",
            "pt":  "merged_pocket_binary_text_comp",
            "ps":  "merged_pocket_sequence_binary_comp",
            "pst": "merged_pocket_sequence_binary_text_comp",
        },
    },
    {
        "panel_title": "ASD + PL8 + Kinase",
        "source": "balanced",
        "tasks": {
            "p":   "ASD_merged_pocket_binary_comp",
            "pt":  "ASD_merged_pocket_binary_text_comp",
            "ps":  "ASD_merged_pocket_sequence_binary_comp",
            "pst": "ASD_merged_pocket_sequence_binary_text_comp",
        },
    },
]


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
    return y_true, y_pred


# ── PR computation ────────────────────────────────────────────────────────────

def extract_scores(y_pred: np.ndarray) -> np.ndarray:
    """
    Convert saved predictions to a 1D score vector.

    Supported input:
      - (N,)
      - (N,1)
      - (N,2) or (N,C): use last column as positive score
    """
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

    # precision_recall_curve returns recall in decreasing order
    recall = recall[::-1]
    precision = precision[::-1]

    aupr = sklearn_auc(recall, precision)
    prevalence = float(np.mean(y_true))
    return recall, precision, aupr, prevalence

def average_pr(load_fn, root_dir: str, model: str, task: str):
    precisions, auprs, prevalences = [], [], []

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


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_figure(base_dir: str, balanced_dir: str, output_dir: str):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.5, 9.0),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for ax, dataset_cfg in zip(axes, DATASETS):
        panel_title = dataset_cfg["panel_title"]
        source = dataset_cfg["source"]
        tasks = dataset_cfg["tasks"]

        missing = []
        baseline_drawn = False

        for emb_key in EMBEDDING_ORDER:
            task = tasks[emb_key]
            color = EMBEDDING_COLORS[emb_key]

            if source == "flat":
                result = average_pr(load_flat, base_dir, MODEL, task)
            elif source == "balanced":
                result = average_pr(load_balanced_hierarchical, balanced_dir, MODEL, task)
            else:
                raise ValueError(f"Unknown source type: {source}")

            if result is None:
                missing.append(task)
                continue

            mean_precision = result["mean_precision"]
            std_precision = result["std_precision"]
            mean_aupr = result["mean_aupr"]
            std_aupr = result["std_aupr"]
            mean_prevalence = result["mean_prevalence"]
            n = result["n"]

            # Draw random-baseline precision once per subplot
            # Use the first successfully loaded task in that subplot
            if not baseline_drawn:
                ax.axhline(
                    y=mean_prevalence,
                    color="gray",
                    linestyle=":",
                    lw=1.0,
                    alpha=0.8,
                    label=f"baseline {mean_prevalence:.2f}",
                )
                baseline_drawn = True

            n_prefix = f"(n={n}) " if n < len(SEEDS) else ""
            label = f"{n_prefix}{EMBEDDING_LABELS[emb_key]}  {mean_aupr:.2f}±{std_aupr:.2f}"

            ax.plot(
                MEAN_RECALL,
                mean_precision,
                color=color,
                lw=2.0,
                label=label,
            )
            ax.fill_between(
                MEAN_RECALL,
                np.clip(mean_precision - std_precision, 0, 1),
                np.clip(mean_precision + std_precision, 0, 1),
                color=color,
                alpha=0.12,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

        if missing:
            print(f"[{panel_title}] Missing tasks: {missing}")

    fig.suptitle(
        f"Representative PR curves for {MODEL_LABEL} across datasets\n"
        f"(balanced ASD split; seeds {SEEDS[0]}–{SEEDS[-1]})",
        fontsize=14,
        fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "figure3_representative_md_no_graph_balancedASD_pr.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
        help="Directory with flat-format PR .npz files.",
    )
    parser.add_argument(
        "--balanced_dir",
        default="/p/scratch/hai_oneprot/curve_plots",
        help="Directory with hierarchical balanced ASD results.",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/figure3_representative_pr",
        help="Output directory for the figure.",
    )
    args = parser.parse_args()

    print(f"Base dir     : {args.base_dir}")
    print(f"Balanced dir : {args.balanced_dir}")
    print(f"Output dir   : {args.output_dir}")
    plot_figure(args.base_dir, args.balanced_dir, args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()