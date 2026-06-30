#!/usr/bin/env python3
"""
plot_supplement_all_models_pr_clean.py

Supplementary PR figure across OneProt encoder architectures.

Also writes an Excel summary table with:
    Encoder, Dataset, Embedding, Mean AUPR, Std AUPR
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc


UNBALANCED_DIR = "<REPO_ROOT>/curve_plots"
BALANCED_DIR = "<CURVE_PLOTS_ROOT>"
OUTPUT_DIR = "<REPO_ROOT>/figure3_imbalance_pr"


MODELS = [
    ("oneprot_pocket_text_32900", "Pocket+Text"),
    ("oneprot_struct_token_pocket_text_32900", "ST+Pocket+Text"),
    ("oneprot_struct_graph_pocket_text_32900", "SG+Pocket+Text"),
    ("oneprot_md_combined_gpcr_no_struct_graph_32900", "MD+ST+Pocket+Text"),
    ("oneprot_md_combined_gpcr_no_struct_token_32900", "MD+SG+Pocket+Text"),
    ("oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity", "ST+SG+Pocket+Text"),
    ("oneprot_md_combined_gpcr_32900", "MD+ST+SG+Pocket+Text"),
]

SEEDS = [11, 12, 13, 14, 15]
MEAN_RECALL = np.linspace(0, 1, 250)


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


ALLODIVERSE_TASKS = {
    "p": "ASD_merged_pocket_binary_comp",
    "pt": "ASD_merged_pocket_binary_text_comp",
    "ps": "ASD_merged_pocket_sequence_binary_comp",
    "pst": "ASD_merged_pocket_sequence_binary_text_comp",
}

KINSITE_TASKS = {
    "p": "Kinase_pocket",
    "pt": "Kinase_pocket_text",
    "ps": "Kinase_combined",
    "pst": "Kinase_combined_text",
}


def _load_npz(path):
    if not os.path.exists(path):
        return None

    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if len(np.unique(y_true)) < 2:
        return None

    return y_true, y_pred


def load_flat(root_dir, model, task, seed):
    path = os.path.join(root_dir, f"{task}_{model}_seed{seed}_preds.npz")
    return _load_npz(path)


def load_balanced_hierarchical(root_dir, model, task, seed):
    path = os.path.join(root_dir, model, f"{task}_balanced", f"seed{seed}", "preds.npz")
    return _load_npz(path)


def extract_scores(y_pred):
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

    return recall, precision, aupr


def average_pr(load_fn, root_dir, model, task):
    precisions = []
    auprs = []

    for seed in SEEDS:
        loaded = load_fn(root_dir, model, task, seed)
        if loaded is None:
            continue

        y_true, y_pred = loaded
        recall, precision, aupr = compute_pr_curve(y_true, y_pred)

        interp_precision = np.interp(MEAN_RECALL, recall, precision)

        precisions.append(interp_precision)
        auprs.append(aupr)

    if not precisions:
        return None

    return {
        "mean_precision": np.mean(precisions, axis=0),
        "std_precision": np.std(precisions, axis=0),
        "mean_aupr": float(np.mean(auprs)),
        "std_aupr": float(np.std(auprs)),
        "n": len(precisions),
    }


def add_curve_with_band(ax, x, y, ystd, color, linestyle):
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        lw=2.0,
        solid_capstyle="round",
    )

    ax.fill_between(
        x,
        np.clip(y - ystd, 0, 1),
        np.clip(y + ystd, 0, 1),
        color=color,
        alpha=0.08,
        linewidth=0,
    )


def style_axis(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(alpha=0.20, linewidth=0.6)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def add_panel(ax, model_name, model_label):
    missing = []
    n_drawn = 0

    for emb in EMBEDDING_ORDER:
        color = EMBEDDING_COLORS[emb]

        allodiverse_task = ALLODIVERSE_TASKS[emb]
        kinsite_task = KINSITE_TASKS[emb]

        res_allodiv_orig = average_pr(
            load_flat,
            UNBALANCED_DIR,
            model_name,
            allodiverse_task,
        )

        res_allodiv_bal = average_pr(
            load_balanced_hierarchical,
            BALANCED_DIR,
            model_name,
            allodiverse_task,
        )

        res_kinsite = average_pr(
            load_flat,
            UNBALANCED_DIR,
            model_name,
            kinsite_task,
        )

        if res_allodiv_orig is not None:
            add_curve_with_band(
                ax,
                MEAN_RECALL,
                res_allodiv_orig["mean_precision"],
                res_allodiv_orig["std_precision"],
                color,
                "-",
            )
            n_drawn += 1
        else:
            missing.append(f"AlloDiverse original / {allodiverse_task}")

        if res_allodiv_bal is not None:
            add_curve_with_band(
                ax,
                MEAN_RECALL,
                res_allodiv_bal["mean_precision"],
                res_allodiv_bal["std_precision"],
                color,
                "--",
            )
            n_drawn += 1
        else:
            missing.append(f"AlloDiverse balanced / {allodiverse_task}")

        if res_kinsite is not None:
            add_curve_with_band(
                ax,
                MEAN_RECALL,
                res_kinsite["mean_precision"],
                res_kinsite["std_precision"],
                color,
                ":",
            )
            n_drawn += 1
        else:
            missing.append(f"KinSite / {kinsite_task}")

    if n_drawn == 0:
        raise RuntimeError(f"No PR curves found for model: {model_name}")

    style_axis(ax, model_label)

    if missing:
        print(f"\n[{model_label}] Missing:")
        for item in missing:
            print(f"  - {item}")


def collect_summary_rows():
    rows = []

    dataset_info = [
        ("AlloDiverse original", load_flat, UNBALANCED_DIR, ALLODIVERSE_TASKS),
        ("AlloDiverse balanced", load_balanced_hierarchical, BALANCED_DIR, ALLODIVERSE_TASKS),
        ("KinSite", load_flat, UNBALANCED_DIR, KINSITE_TASKS),
    ]

    for model_name, model_label in MODELS:
        for dataset_label, loader, root_dir, task_dict in dataset_info:
            for emb in EMBEDDING_ORDER:
                result = average_pr(
                    loader,
                    root_dir,
                    model_name,
                    task_dict[emb],
                )

                if result is None:
                    print(
                        f"Missing summary row: "
                        f"{model_label} | {dataset_label} | {EMBEDDING_LABELS[emb]}"
                    )
                    continue

                rows.append({
                    "Encoder": model_label,
                    "Dataset": dataset_label,
                    "Embedding": EMBEDDING_LABELS[emb],
                    "Mean AUPR": result["mean_aupr"],
                    "Std AUPR": result["std_aupr"],
                })

    return rows


def add_shared_legend(legend_ax):
    legend_ax.axis("off")

    embedding_handles = [
        Line2D(
            [0],
            [0],
            color=EMBEDDING_COLORS[emb],
            lw=3.0,
            label=EMBEDDING_LABELS[emb],
        )
        for emb in EMBEDDING_ORDER
    ]

    condition_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.5,
            linestyle="-",
            label="AlloDiverse, original training",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.5,
            linestyle="--",
            label="AlloDiverse, balanced training",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.5,
            linestyle=":",
            label="KinSite",
        ),
    ]

    legend1 = legend_ax.legend(
        handles=embedding_handles,
        title="Embedding combination",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.93),
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        borderpad=0.8,
        labelspacing=0.7,
        handlelength=2.8,
    )

    legend_ax.add_artist(legend1)

    legend2 = legend_ax.legend(
        handles=condition_handles,
        title="Dataset / training condition",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.46),
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        borderpad=0.8,
        labelspacing=0.7,
        handlelength=2.8,
    )

    legend_ax.add_artist(legend2)

    legend_ax.text(
        0.02,
        0.01,
        "Shaded bands indicate variability\nacross available independent runs.",
        fontsize=9.5,
        ha="left",
        va="bottom",
        transform=legend_ax.transAxes,
    )


def plot_figure():
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(22, 10.5),
        sharex=True,
        sharey=True,
    )

    axes_flat = axes.flatten()

    for i, (model_name, model_label) in enumerate(MODELS):
        add_panel(axes_flat[i], model_name, model_label)

    add_shared_legend(axes_flat[-1])

    fig.suptitle(
        "Precision–recall behaviour across OneProt encoder architectures",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.948,
        "Color denotes extracted embedding combination; line style denotes dataset or training condition.",
        ha="center",
        va="center",
        fontsize=12,
    )

    plt.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.895,
        wspace=0.11,
        hspace=0.30,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_png = os.path.join(OUTPUT_DIR, "supplement_all_models_pr_clean.png")
    out_pdf = os.path.join(OUTPUT_DIR, "supplement_all_models_pr_clean.pdf")
    out_xlsx = os.path.join(OUTPUT_DIR, "supplement_all_models_pr_summary.xlsx")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    summary_rows = collect_summary_rows()
    df = pd.DataFrame(summary_rows)
    df.to_excel(out_xlsx, index=False)

    print(f"\nSaved PNG:  {out_png}")
    print(f"Saved PDF:  {out_pdf}")
    print(f"Saved XLSX: {out_xlsx}")


def main():
    print(f"Unbalanced dir: {UNBALANCED_DIR}")
    print(f"Balanced dir:   {BALANCED_DIR}")
    print(f"Output dir:     {OUTPUT_DIR}")

    plot_figure()
    print("Done.")


if __name__ == "__main__":
    main()