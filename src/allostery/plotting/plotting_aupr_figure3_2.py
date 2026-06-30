"""
plot_figure3_two_models_pr.py

Figure 3: Precision-Recall comparison for two representative models.

Design:
    - 1x2 panels = one panel per model
    - Each panel shows:
        * ASD + PL8 + Kinase (unbalanced)
        * ASD + PL8 + Kinase (balanced)
        * Kinase (unbalanced)
    - 4 embedding sets:
        pocket
        pocket+text
        pocket+sequence
        pocket+sequence+text
    - same color = same embedding
    - solid   = ASD unbalanced
    - dashed  = ASD balanced
    - dotted  = Kinase
    - AUPR shown in the legend
    - legend below each subplot

Data sources:
    - Unbalanced flat:
          <unbalanced_dir>/{task}_{model}_seed{seed}_preds.npz
    - Balanced hierarchical:
          <balanced_dir>/<model>/{task}_balanced/seed{seed}/preds.npz

Usage:
    python plot_figure3_two_models_pr.py \
        --model1 oneprot_md_combined_gpcr_no_struct_graph_32900 \
        --label1 md_no_graph \
        --model2 EXACT_SECOND_MODEL_NAME \
        --label2 second_model \
        --unbalanced_dir <REPO_ROOT>/curve_plots \
        --balanced_dir <CURVE_PLOTS_ROOT> \
        --output_dir <REPO_ROOT>/figure3_imbalance_pr
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc


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
    "p":   "#1f77b4",
    "pt":  "#ff7f0e",
    "ps":  "#2ca02c",
    "pst": "#d62728",
}

ASD_TASKS = {
    "p":   "ASD_merged_pocket_binary_comp",
    "pt":  "ASD_merged_pocket_binary_text_comp",
    "ps":  "ASD_merged_pocket_sequence_binary_comp",
    "pst": "ASD_merged_pocket_sequence_binary_text_comp",
}

KINASE_TASKS = {
    "p":   "Kinase_pocket",
    "pt":  "Kinase_pocket_text",
    "ps":  "Kinase_combined",
    "pst": "Kinase_combined_text",
}


def load_flat(unbalanced_dir: str, model: str, task: str, seed: int):
    path = os.path.join(unbalanced_dir, f"{task}_{model}_seed{seed}_preds.npz")
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

def extract_scores(y_pred: np.ndarray) -> np.ndarray:
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

def style_axis(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)

def add_curve_with_band(ax, x, y, ystd, color, linestyle, label):
    ax.plot(x, y, color=color, lw=2.0, linestyle=linestyle, label=label)
    ax.fill_between(
        x,
        np.clip(y - ystd, 0, 1),
        np.clip(y + ystd, 0, 1),
        color=color,
        alpha=0.10,
    )

def add_panel(ax, unbalanced_dir, balanced_dir, model_name, model_label):
    baseline_vals = []
    missing = []
    n_drawn = 0

    for emb in EMBEDDING_ORDER:
        color = EMBEDDING_COLORS[emb]
        emb_label = EMBEDDING_LABELS[emb]

        asd_task = ASD_TASKS[emb]
        kinase_task = KINASE_TASKS[emb]

        res_asd_unbal = average_pr(load_flat, unbalanced_dir, model_name, asd_task)
        res_asd_bal   = average_pr(load_balanced_hierarchical, balanced_dir, model_name, asd_task)
        res_kinase    = average_pr(load_flat, unbalanced_dir, model_name, kinase_task)

        if res_asd_unbal is None:
            missing.append(f"unbal/{asd_task}")
        else:
            add_curve_with_band(
                ax=ax,
                x=MEAN_RECALL,
                y=res_asd_unbal["mean_precision"],
                ystd=res_asd_unbal["std_precision"],
                color=color,
                linestyle="-",
                label=f"{emb_label} ASD unbal  {res_asd_unbal['mean_aupr']:.2f}",
            )
            baseline_vals.append(res_asd_unbal["mean_prevalence"])
            n_drawn += 1

        if res_asd_bal is None:
            missing.append(f"bal/{asd_task}")
        else:
            add_curve_with_band(
                ax=ax,
                x=MEAN_RECALL,
                y=res_asd_bal["mean_precision"],
                ystd=res_asd_bal["std_precision"],
                color=color,
                linestyle="--",
                label=f"{emb_label} ASD bal  {res_asd_bal['mean_aupr']:.2f}",
            )
            baseline_vals.append(res_asd_bal["mean_prevalence"])
            n_drawn += 1

        if res_kinase is None:
            missing.append(f"kin/{kinase_task}")
        else:
            add_curve_with_band(
                ax=ax,
                x=MEAN_RECALL,
                y=res_kinase["mean_precision"],
                ystd=res_kinase["std_precision"],
                color=color,
                linestyle=":",
                label=f"{emb_label} Kinase  {res_kinase['mean_aupr']:.2f}",
            )
            baseline_vals.append(res_kinase["mean_prevalence"])
            n_drawn += 1

    if n_drawn == 0:
        raise RuntimeError(
            f"No curves found for model '{model_name}'.\n"
            f"Check the exact model name in both:\n"
            f"  flat files: <unbalanced_dir>/{{task}}_{model_name}_seed{{seed}}_preds.npz\n"
            f"  balanced:   <balanced_dir>/{model_name}/{{task}}_balanced/seed{{seed}}/preds.npz"
        )

    if baseline_vals:
        ax.axhline(
            y=float(np.mean(baseline_vals)),
            color="gray",
            linestyle=(0, (1, 2)),
            lw=1.0,
            alpha=0.9,
            label=f"baseline {float(np.mean(baseline_vals)):.2f}",
        )

    style_axis(ax, model_label)

    ax.legend(
        fontsize=7.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        framealpha=0.95,
    )

    if missing:
        print(f"[{model_label}] Missing tasks:")
        for m in missing:
            print(f"  - {m}")

def plot_figure(model_specs, unbalanced_dir: str, balanced_dir: str, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2))

    for ax, model_spec in zip(axes, model_specs):
        add_panel(
            ax=ax,
            unbalanced_dir=unbalanced_dir,
            balanced_dir=balanced_dir,
            model_name=model_spec["name"],
            model_label=model_spec["label"],
        )

    fig.suptitle(
        f"Imbalance-focused PR curves\n"
        f"(ASD balanced vs unbalanced; Kinase unbalanced; seeds {SEEDS[0]}–{SEEDS[-1]})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.92])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "figure3_two_models_pr.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", required=True, help="Exact first model name.")
    parser.add_argument("--label1", default="model1", help="Display label for first model.")
    parser.add_argument("--model2", required=True, help="Exact second model name.")
    parser.add_argument("--label2", default="model2", help="Display label for second model.")
    parser.add_argument(
        "--unbalanced_dir",
        default="<REPO_ROOT>/curve_plots",
        help="Directory with flat-format prediction .npz files.",
    )
    parser.add_argument(
        "--balanced_dir",
        default="<CURVE_PLOTS_ROOT>",
        help="Directory with hierarchical balanced ASD results.",
    )
    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/figure3_imbalance_pr",
        help="Output directory for the figure.",
    )
    args = parser.parse_args()

    model_specs = [
        {"name": args.model1, "label": args.label1},
        {"name": args.model2, "label": args.label2},
    ]

    print(f"Model 1        : {args.model1}")
    print(f"Model 2        : {args.model2}")
    print(f"Unbalanced dir : {args.unbalanced_dir}")
    print(f"Balanced dir   : {args.balanced_dir}")
    print(f"Output dir     : {args.output_dir}")

    plot_figure(model_specs, args.unbalanced_dir, args.balanced_dir, args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()