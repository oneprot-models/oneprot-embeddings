"""
plot_figure3_tpr_tnr_avg_models_fixed.py

Figure 3 (averaged across models):
Threshold-dependent behaviour averaged over model architectures.

Design:
    - all 7 models
    - 2x2 panels:
        (1) ASD+PL8+Kinase  : TPR, balanced vs unbalanced
        (2) ASD+PL8+Kinase  : TNR, balanced vs unbalanced
        (3) Kinase          : TPR
        (4) Kinase          : TNR
    - x-axis = embedding sets:
        pocket
        pocket+text
        pocket+sequence
        pocket+sequence+text
    - bars = mean across models
    - error bars = std across models
    - points = individual models
    - balanced ASD results loaded from hierarchical balanced folder
    - unbalanced ASD + Kinase loaded from flat files

This version is robust to:
    - y_pred being probabilities or logits
    - y_pred being 1D or 2D
    - optional inversion of the score direction

Usage examples:

1) Debug a few files first:
    python plot_figure3_tpr_tnr_avg_models_fixed.py \
        --debug_predictions

2) If y_pred are probabilities in [0,1]:
    python plot_figure3_tpr_tnr_avg_models_fixed.py \
        --score_type prob \
        --threshold 0.5

3) If y_pred are logits / raw scores:
    python plot_figure3_tpr_tnr_avg_models_fixed.py \
        --score_type logit

4) If the positive class score is reversed:
    python plot_figure3_tpr_tnr_avg_models_fixed.py \
        --score_type prob \
        --invert_scores
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]
EMBEDDING_LABELS = {
    "p": "pocket",
    "pt": "pocket+text",
    "ps": "pocket+sequence",
    "pst": "pocket+sequence+text",
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

COLOR_UNBAL = "#cc4c4c"
COLOR_BAL = "#4c72b0"
COLOR_KIN = "#7a5aa6"
SCATTER_COLOR = "#333333"


# ── Loading ───────────────────────────────────────────────────────────────────

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


# ── Score handling ────────────────────────────────────────────────────────────

def infer_score_type(y_pred: np.ndarray) -> str:
    arr = np.asarray(y_pred)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "prob"
    mn, mx = float(arr.min()), float(arr.max())
    if mn >= 0.0 and mx <= 1.0:
        return "prob"
    return "logit"

def extract_positive_scores(y_pred: np.ndarray, positive_index: int = -1) -> np.ndarray:
    """
    Return a 1D score vector for the positive class.

    Supported input shapes:
      - (N,)
      - (N,1)
      - (N,2) or (N,C): uses selected column
    """
    arr = np.asarray(y_pred)

    if arr.ndim == 1:
        return arr.astype(float)

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        return arr[:, positive_index].astype(float)

    raise ValueError(f"Unsupported y_pred shape: {arr.shape}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_tpr_tnr(
    y_true,
    y_pred,
    threshold=0.5,
    score_type="auto",
    invert_scores=False,
    positive_index=-1,
):
    """
    Compute TPR/TNR after converting predictions to binary labels.

    score_type:
        - "auto": infer from value range
        - "prob": treat scores as probabilities in [0,1]
        - "logit": treat scores as logits/raw scores, threshold at 0.0
    """
    y_true = np.asarray(y_true).astype(int)
    scores = extract_positive_scores(y_pred, positive_index=positive_index)

    if score_type == "auto":
        resolved_score_type = infer_score_type(scores)
    else:
        resolved_score_type = score_type

    if invert_scores:
        if resolved_score_type == "prob":
            scores = 1.0 - scores
        else:
            scores = -scores

    if resolved_score_type == "prob":
        y_hat = (scores >= threshold).astype(int)
    elif resolved_score_type == "logit":
        y_hat = (scores >= 0.0).astype(int)
    else:
        raise ValueError(f"Unknown score_type: {resolved_score_type}")

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    tp = np.sum((y_hat == 1) & pos_mask)
    fn = np.sum((y_hat == 0) & pos_mask)
    tn = np.sum((y_hat == 0) & neg_mask)
    fp = np.sum((y_hat == 1) & neg_mask)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return tpr, tnr


def average_over_seeds(
    load_fn,
    root_dir: str,
    model: str,
    task: str,
    threshold=0.5,
    score_type="auto",
    invert_scores=False,
    positive_index=-1,
):
    tprs, tnrs = [], []

    for seed in SEEDS:
        loaded = load_fn(root_dir, model, task, seed)
        if loaded is None:
            continue

        y_true, y_pred = loaded
        tpr, tnr = compute_tpr_tnr(
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            score_type=score_type,
            invert_scores=invert_scores,
            positive_index=positive_index,
        )

        if not np.isnan(tpr):
            tprs.append(tpr)
        if not np.isnan(tnr):
            tnrs.append(tnr)

    if len(tprs) == 0 or len(tnrs) == 0:
        return None

    return {
        "mean_tpr": float(np.mean(tprs)),
        "mean_tnr": float(np.mean(tnrs)),
        "std_tpr_seed": float(np.std(tprs)),
        "std_tnr_seed": float(np.std(tnrs)),
        "n": min(len(tprs), len(tnrs)),
    }


# ── Debugging ─────────────────────────────────────────────────────────────────

def debug_predictions(unbalanced_dir, balanced_dir, positive_index=-1):
    print("\n=== DEBUGGING PREDICTION RANGES ===\n")

    examples = [
        ("Kinase", "flat", KINASE_TASKS["p"], MODELS[2], 11),
        ("Kinase", "flat", KINASE_TASKS["pt"], MODELS[2], 11),
        ("ASD", "flat", ASD_TASKS["pst"], MODELS[2], 11),
        ("ASD_bal", "balanced", ASD_TASKS["pst"], MODELS[2], 11),
    ]

    for name, source, task, model, seed in examples:
        if source == "flat":
            loaded = load_flat(unbalanced_dir, model, task, seed)
            path = os.path.join(unbalanced_dir, f"{task}_{model}_seed{seed}_preds.npz")
        else:
            loaded = load_balanced_hierarchical(balanced_dir, model, task, seed)
            path = os.path.join(balanced_dir, model, f"{task}_balanced", f"seed{seed}", "preds.npz")

        print(f"\n[{name}] {path}")
        if loaded is None:
            print("  MISSING")
            continue

        y_true, y_pred = loaded
        scores = extract_positive_scores(y_pred, positive_index=positive_index)
        inferred = infer_score_type(scores)

        uniq, counts = np.unique(y_true, return_counts=True)
        print(f"  y_true counts: {dict(zip(uniq.tolist(), counts.tolist()))}")
        print(f"  y_pred shape : {np.asarray(y_pred).shape}")
        print(f"  scores min/max/mean: {scores.min():.4f} / {scores.max():.4f} / {scores.mean():.4f}")
        print(f"  inferred score_type: {inferred}")
        print(f"  first 10 scores: {np.round(scores[:10], 4)}")


# ── Aggregation across models ─────────────────────────────────────────────────

def collect_metric_by_model(
    unbalanced_dir,
    balanced_dir,
    threshold,
    score_type,
    invert_scores,
    positive_index,
):
    results = {
        "asd_unbal_tpr": {emb: [] for emb in EMBEDDING_ORDER},
        "asd_bal_tpr":   {emb: [] for emb in EMBEDDING_ORDER},
        "asd_unbal_tnr": {emb: [] for emb in EMBEDDING_ORDER},
        "asd_bal_tnr":   {emb: [] for emb in EMBEDDING_ORDER},
        "kin_tpr":       {emb: [] for emb in EMBEDDING_ORDER},
        "kin_tnr":       {emb: [] for emb in EMBEDDING_ORDER},
    }

    for model in MODELS:
        for emb in EMBEDDING_ORDER:
            task_asd = ASD_TASKS[emb]
            res_asd_unbal = average_over_seeds(
                load_flat, unbalanced_dir, model, task_asd,
                threshold=threshold,
                score_type=score_type,
                invert_scores=invert_scores,
                positive_index=positive_index,
            )
            res_asd_bal = average_over_seeds(
                load_balanced_hierarchical, balanced_dir, model, task_asd,
                threshold=threshold,
                score_type=score_type,
                invert_scores=invert_scores,
                positive_index=positive_index,
            )

            if res_asd_unbal is not None:
                results["asd_unbal_tpr"][emb].append(res_asd_unbal["mean_tpr"])
                results["asd_unbal_tnr"][emb].append(res_asd_unbal["mean_tnr"])

            if res_asd_bal is not None:
                results["asd_bal_tpr"][emb].append(res_asd_bal["mean_tpr"])
                results["asd_bal_tnr"][emb].append(res_asd_bal["mean_tnr"])

            task_kin = KINASE_TASKS[emb]
            res_kin = average_over_seeds(
                load_flat, unbalanced_dir, model, task_kin,
                threshold=threshold,
                score_type=score_type,
                invert_scores=invert_scores,
                positive_index=positive_index,
            )

            if res_kin is not None:
                results["kin_tpr"][emb].append(res_kin["mean_tpr"])
                results["kin_tnr"][emb].append(res_kin["mean_tnr"])

    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────

def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))

def add_scatter(ax, xpos, values, jitter=0.05, alpha=0.75):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return
    jitter_offsets = np.linspace(-jitter, jitter, len(values)) if len(values) > 1 else np.array([0.0])
    ax.scatter(
        np.full(len(values), xpos) + jitter_offsets,
        values,
        color=SCATTER_COLOR,
        s=22,
        alpha=alpha,
        zorder=3,
        linewidths=0.3,
        edgecolors="white",
    )

def add_bar_labels(ax, bars, values):
    for rect, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

def style_axis(ax, ylabel, title):
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=20, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

def summarize(metric_dict):
    means = []
    stds = []
    for emb in EMBEDDING_ORDER:
        m, s = mean_std(metric_dict[emb])
        means.append(m)
        stds.append(s)
    return means, stds


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_figure(
    unbalanced_dir: str,
    balanced_dir: str,
    output_dir: str,
    threshold: float,
    score_type: str,
    invert_scores: bool,
    positive_index: int,
):
    res = collect_metric_by_model(
        unbalanced_dir=unbalanced_dir,
        balanced_dir=balanced_dir,
        threshold=threshold,
        score_type=score_type,
        invert_scores=invert_scores,
        positive_index=positive_index,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), constrained_layout=True)
    ax_asd_tpr, ax_asd_tnr, ax_kin_tpr, ax_kin_tnr = axes.flatten()

    x = np.arange(len(EMBEDDING_ORDER))
    width = 0.36

    # ASD TPR
    asd_tpr_unbal_mean, asd_tpr_unbal_std = summarize(res["asd_unbal_tpr"])
    asd_tpr_bal_mean, asd_tpr_bal_std = summarize(res["asd_bal_tpr"])

    bars1 = ax_asd_tpr.bar(
        x - width/2, asd_tpr_unbal_mean, width,
        yerr=asd_tpr_unbal_std, capsize=4,
        color=COLOR_UNBAL, label="unbalanced", alpha=0.9
    )
    bars2 = ax_asd_tpr.bar(
        x + width/2, asd_tpr_bal_mean, width,
        yerr=asd_tpr_bal_std, capsize=4,
        color=COLOR_BAL, label="balanced", alpha=0.9
    )
    for i, emb in enumerate(EMBEDDING_ORDER):
        add_scatter(ax_asd_tpr, x[i] - width/2, res["asd_unbal_tpr"][emb])
        add_scatter(ax_asd_tpr, x[i] + width/2, res["asd_bal_tpr"][emb])
    ax_asd_tpr.set_xticks(x)
    ax_asd_tpr.set_xticklabels([EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER])
    style_axis(ax_asd_tpr, "True positive rate", "ASD + PL8 + Kinase")
    ax_asd_tpr.legend(fontsize=9, framealpha=0.9)
    add_bar_labels(ax_asd_tpr, bars1, asd_tpr_unbal_mean)
    add_bar_labels(ax_asd_tpr, bars2, asd_tpr_bal_mean)

    # ASD TNR
    asd_tnr_unbal_mean, asd_tnr_unbal_std = summarize(res["asd_unbal_tnr"])
    asd_tnr_bal_mean, asd_tnr_bal_std = summarize(res["asd_bal_tnr"])

    bars3 = ax_asd_tnr.bar(
        x - width/2, asd_tnr_unbal_mean, width,
        yerr=asd_tnr_unbal_std, capsize=4,
        color=COLOR_UNBAL, label="unbalanced", alpha=0.9
    )
    bars4 = ax_asd_tnr.bar(
        x + width/2, asd_tnr_bal_mean, width,
        yerr=asd_tnr_bal_std, capsize=4,
        color=COLOR_BAL, label="balanced", alpha=0.9
    )
    for i, emb in enumerate(EMBEDDING_ORDER):
        add_scatter(ax_asd_tnr, x[i] - width/2, res["asd_unbal_tnr"][emb])
        add_scatter(ax_asd_tnr, x[i] + width/2, res["asd_bal_tnr"][emb])
    ax_asd_tnr.set_xticks(x)
    ax_asd_tnr.set_xticklabels([EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER])
    style_axis(ax_asd_tnr, "True negative rate", "ASD + PL8 + Kinase")
    ax_asd_tnr.legend(fontsize=9, framealpha=0.9)
    add_bar_labels(ax_asd_tnr, bars3, asd_tnr_unbal_mean)
    add_bar_labels(ax_asd_tnr, bars4, asd_tnr_bal_mean)

    # Kinase TPR
    kin_tpr_mean, kin_tpr_std = summarize(res["kin_tpr"])
    bars5 = ax_kin_tpr.bar(
        x, kin_tpr_mean, width=0.55,
        yerr=kin_tpr_std, capsize=4,
        color=COLOR_KIN, alpha=0.9
    )
    for i, emb in enumerate(EMBEDDING_ORDER):
        add_scatter(ax_kin_tpr, x[i], res["kin_tpr"][emb])
    ax_kin_tpr.set_xticks(x)
    ax_kin_tpr.set_xticklabels([EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER])
    style_axis(ax_kin_tpr, "True positive rate", "Kinase")
    add_bar_labels(ax_kin_tpr, bars5, kin_tpr_mean)

    # Kinase TNR
    kin_tnr_mean, kin_tnr_std = summarize(res["kin_tnr"])
    bars6 = ax_kin_tnr.bar(
        x, kin_tnr_mean, width=0.55,
        yerr=kin_tnr_std, capsize=4,
        color=COLOR_KIN, alpha=0.9
    )
    for i, emb in enumerate(EMBEDDING_ORDER):
        add_scatter(ax_kin_tnr, x[i], res["kin_tnr"][emb])
    ax_kin_tnr.set_xticks(x)
    ax_kin_tnr.set_xticklabels([EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER])
    style_axis(ax_kin_tnr, "True negative rate", "Kinase")
    add_bar_labels(ax_kin_tnr, bars6, kin_tnr_mean)

    threshold_text = f"threshold={threshold}" if score_type != "logit" else "threshold=0.0 (logit)"
    fig.suptitle(
        f"Threshold-dependent behaviour averaged across models\n"
        f"Bars show mean across models, error bars show SD across models, points show individual models\n"
        f"score_type={score_type}, invert_scores={invert_scores}, positive_index={positive_index}, {threshold_text}",
        fontsize=12,
        fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "figure3_tpr_tnr_avg_models_fixed.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unbalanced_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
    )
    parser.add_argument(
        "--balanced_dir",
        default="/p/scratch/hai_oneprot/curve_plots",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/figure3_tpr_tnr_avg_models",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Used only when score_type=prob.",
    )
    parser.add_argument(
        "--score_type",
        default="auto",
        choices=["auto", "prob", "logit"],
        help="How to interpret y_pred.",
    )
    parser.add_argument(
        "--invert_scores",
        action="store_true",
        help="Flip score direction if the positive class is reversed.",
    )
    parser.add_argument(
        "--positive_index",
        type=int,
        default=-1,
        help="Which column to use when y_pred is 2D. Default: last column.",
    )
    parser.add_argument(
        "--debug_predictions",
        action="store_true",
        help="Print a few prediction ranges and inferred score types, then continue.",
    )
    args = parser.parse_args()

    print(f"Unbalanced dir : {args.unbalanced_dir}")
    print(f"Balanced dir   : {args.balanced_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Threshold      : {args.threshold}")
    print(f"Score type     : {args.score_type}")
    print(f"Invert scores  : {args.invert_scores}")
    print(f"Positive index : {args.positive_index}")

    if args.debug_predictions:
        debug_predictions(
            unbalanced_dir=args.unbalanced_dir,
            balanced_dir=args.balanced_dir,
            positive_index=args.positive_index,
        )

    plot_figure(
        unbalanced_dir=args.unbalanced_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        score_type=args.score_type,
        invert_scores=args.invert_scores,
        positive_index=args.positive_index,
    )
    print("Done.")


if __name__ == "__main__":
    main()