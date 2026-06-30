"""
plot_supplement_all_models_pr.py

Supplementary figure: PR curves for all 7 models.

- 2x4 grid (last panel empty)
- Each panel:
    ASD unbalanced (solid)
    ASD balanced   (dashed)
    Kinase         (dotted)
- Same colors = same embedding set
- Uses the exact same file/path logic as the old working script
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc


# ── Hardcoded directories (same style as old script) ─────────────────────────

UNBALANCED_DIR = "/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots"
BALANCED_DIR   = "/p/scratch/hai_oneprot/curve_plots"
OUTPUT_DIR     = "/p/project1/hai_oneprot/bazarova1/oneprot-panda/figure3_imbalance_pr"


# ── Hardcoded models ──────────────────────────────────────────────────────────

MODELS = [
    ("oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity", "ST+SG+Pocket+Text"),
    ("oneprot_md_combined_gpcr_32900", "MD+ST+SG+Pocket+Text"),
    ("oneprot_md_combined_gpcr_no_struct_graph_32900", "MD+ST+Pocket+Text"),
    ("oneprot_md_combined_gpcr_no_struct_token_32900", "MD+SG+Pocket+Text"),
    ("oneprot_pocket_text_32900", "Pocket+Text"),
    ("oneprot_struct_graph_pocket_text_32900", "SG+Pocket+Text"),
    ("oneprot_struct_token_pocket_text_32900", "ST+Pocket+Text"),
]


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


# ── Loading helpers (unchanged path logic) ───────────────────────────────────

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


# ── PR helpers ────────────────────────────────────────────────────────────────

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

    # make recall increasing
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


# ── Plot helpers ──────────────────────────────────────────────────────────────

def add_curve_with_band(ax, x, y, ystd, color, linestyle, label):
    ax.plot(x, y, color=color, lw=1.8, linestyle=linestyle, label=label)
    ax.fill_between(
        x,
        np.clip(y - ystd, 0, 1),
        np.clip(y + ystd, 0, 1),
        color=color,
        alpha=0.08,
    )

def style_axis(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

def add_panel(ax, model_name, model_label):
    baseline_vals = []
    missing = []
    n_drawn = 0

    for emb in EMBEDDING_ORDER:
        color = EMBEDDING_COLORS[emb]
        emb_label = EMBEDDING_LABELS[emb]

        asd_task = ASD_TASKS[emb]
        kinase_task = KINASE_TASKS[emb]

        res_asd_unbal = average_pr(load_flat, UNBALANCED_DIR, model_name, asd_task)
        res_asd_bal   = average_pr(load_balanced_hierarchical, BALANCED_DIR, model_name, asd_task)
        res_kinase    = average_pr(load_flat, UNBALANCED_DIR, model_name, kinase_task)

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
                label=f"{emb_label} ASD unbal {res_asd_unbal['mean_aupr']:.2f}",
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
                label=f"{emb_label} ASD bal {res_asd_bal['mean_aupr']:.2f}",
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
                label=f"{emb_label} Kinase {res_kinase['mean_aupr']:.2f}",
            )
            baseline_vals.append(res_kinase["mean_prevalence"])
            n_drawn += 1

    if n_drawn == 0:
        raise RuntimeError(
            f"No curves found for model '{model_name}'. "
            f"Check files under:\n"
            f"  {UNBALANCED_DIR}\n"
            f"  {BALANCED_DIR}"
        )

    if baseline_vals:
        ax.axhline(
            y=float(np.mean(baseline_vals)),
            color="gray",
            linestyle=(0, (1, 2)),
            lw=1.0,
            alpha=0.9,
        )

    style_axis(ax, model_label)

    ax.legend(
        fontsize=6.4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        framealpha=0.95,
    )

    if missing:
        print(f"[{model_label}] Missing tasks:")
        for m in missing:
            print(f"  - {m}")


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_figure():
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (model_name, model_label) in enumerate(MODELS):
        add_panel(axes[i], model_name, model_label)

    # hide final empty panel
    for j in range(len(MODELS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Supplementary PR curves across all models\n"
        f"(ASD balanced vs unbalanced; Kinase unbalanced; seeds {SEEDS[0]}–{SEEDS[-1]})",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "supplement_all_models_pr.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


def main():
    print(f"Unbalanced dir : {UNBALANCED_DIR}")
    print(f"Balanced dir   : {BALANCED_DIR}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print("Models:")
    for model_name, model_label in MODELS:
        print(f"  - {model_label}: {model_name}")

    plot_figure()
    print("Done.")


if __name__ == "__main__":
    main()