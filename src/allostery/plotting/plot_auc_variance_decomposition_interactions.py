"""
plot_auc_variance_decomposition_all_pairwise.py

Variance decomposition of ROC-AUC values with all requested pairwise interactions.

Model:
    auc ~ Dataset + Embedding + Architecture + Seed
          + Dataset:Embedding
          + Dataset:Architecture
          + Dataset:Seed
          + Embedding:Architecture
          + Embedding:Seed
          + Architecture:Seed

Usage:
    python plot_auc_variance_decomposition_all_pairwise.py \
        --base_dir <REPO_ROOT>/curve_plots \
        --balanced_dir <CURVE_PLOTS_ROOT> \
        --output_dir <REPO_ROOT>/separability_figures
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc as sklearn_auc
import statsmodels.api as sm
from statsmodels.formula.api import ols


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

DATASET_ORDER = [
    "PL8",
    "Kinase",
    "PL8 + Kinase",
    "ASD + PL8 + Kinase",
]

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket",
    "pt": "Pocket+Text",
    "ps": "Pocket+Sequence",
    "pst": "Pocket+Sequence+Text",
}

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


def load_auc(path):
    if not os.path.exists(path):
        return None

    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(sklearn_auc(fpr, tpr))


def get_path(base_dir, balanced_dir, dataset_name, embedding_key, model, seed):
    task = TASK_MAP[embedding_key][dataset_name]

    if dataset_name == "ASD + PL8 + Kinase":
        return os.path.join(
            balanced_dir,
            model,
            f"{task}_balanced",
            f"seed{seed}",
            "preds.npz",
        )

    return os.path.join(
        base_dir,
        f"{task}_{model}_seed{seed}_preds.npz",
    )


def collect_auc_dataframe(base_dir, balanced_dir):
    rows = []

    for dataset_name in DATASET_ORDER:
        for embedding_key in EMBEDDING_ORDER:
            for model in MODELS:
                for seed in SEEDS:
                    path = get_path(
                        base_dir,
                        balanced_dir,
                        dataset_name,
                        embedding_key,
                        model,
                        seed,
                    )

                    auc_value = load_auc(path)

                    if auc_value is None:
                        continue

                    rows.append(
                        {
                            "auc": auc_value,
                            "dataset": dataset_name,
                            "embedding": EMBEDDING_LABELS[embedding_key],
                            "architecture": MODEL_LABELS[model],
                            "seed": str(seed),
                        }
                    )

    return pd.DataFrame(rows)


def build_formula():
    return (
        "auc ~ "
        "C(dataset) + C(embedding) + C(architecture) + C(seed) + "
        "C(dataset):C(embedding) + "
        "C(dataset):C(architecture) + "
        #"C(dataset):C(seed) + "
        "C(embedding):C(architecture)"
        #"C(embedding):C(seed) + "
        #"C(architecture):C(seed)"
    )


def variance_decomposition(df):
    formula = build_formula()

    fitted_model = ols(formula, data=df).fit()
    anova = sm.stats.anova_lm(fitted_model, typ=2)

    terms = {
        "Dataset": "C(dataset)",
        "Embedding": "C(embedding)",
        "Architecture": "C(architecture)",
        "Seed": "C(seed)",
        "Dataset x Embedding": "C(dataset):C(embedding)",
        "Dataset x Architecture": "C(dataset):C(architecture)",
        #"Dataset x Seed": "C(dataset):C(seed)",
        "Embedding x Architecture": "C(embedding):C(architecture)",
        #"Embedding x Seed": "C(embedding):C(seed)",
        #"Architecture x Seed": "C(architecture):C(seed)",
        "Residual": "Residual",
    }

    values = []

    for label, term in terms.items():
        if term in anova.index:
            values.append((label, float(anova.loc[term, "sum_sq"])))
        else:
            values.append((label, 0.0))

    result = pd.DataFrame(values, columns=["factor", "sum_sq"])
    total = result["sum_sq"].sum()

    result["fraction"] = result["sum_sq"] / total
    result["percent"] = 100 * result["fraction"]

    return result, anova, fitted_model, formula


def plot_variance_decomposition(base_dir, balanced_dir, output_dir):
    df = collect_auc_dataframe(base_dir, balanced_dir)

    if df.empty:
        raise RuntimeError("No AUC values loaded. Check input directories.")

    result, anova, fitted_model, formula = variance_decomposition(df)

    os.makedirs(output_dir, exist_ok=True)

    suffix = "all_pairwise_interactions"

    out_csv = os.path.join(
        output_dir,
        f"auc_variance_decomposition_{suffix}.csv",
    )
    out_anova = os.path.join(
        output_dir,
        f"auc_variance_decomposition_{suffix}_anova.csv",
    )
    out_data = os.path.join(
        output_dir,
        f"auc_variance_decomposition_{suffix}_raw_auc_values.csv",
    )

    result.to_csv(out_csv, index=False)
    anova.to_csv(out_anova)
    df.to_csv(out_data, index=False)

    fig, ax = plt.subplots(figsize=(12.5, 5.6))

    ax.bar(result["factor"], result["percent"])

    ax.set_ylabel("Variance explained (%)")
    ax.set_title(
        "Variance decomposition of ROC-AUC performance\n"
        "main effects and all pairwise interactions",
        fontsize=13,
        fontweight="bold",
    )

    ymax = max(100, result["percent"].max() * 1.2)
    ax.set_ylim(0, ymax)

    ax.tick_params(axis="x", rotation=30)

    for i, value in enumerate(result["percent"]):
        ax.text(
            i,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_png = os.path.join(
        output_dir,
        f"auc_variance_decomposition_{suffix}.png",
    )
    out_pdf = os.path.join(
        output_dir,
        f"auc_variance_decomposition_{suffix}.pdf",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nFormula:")
    print(formula)

    print("\nLoaded observations by dataset and embedding:")
    print(df.groupby(["dataset", "embedding"]).size())

    print("\nANOVA:")
    print(anova)

    print("\nVariance decomposition:")
    print(result)

    print(f"\nSaved PNG      -> {out_png}")
    print(f"Saved PDF      -> {out_pdf}")
    print(f"Saved summary  -> {out_csv}")
    print(f"Saved ANOVA    -> {out_anova}")
    print(f"Saved raw data -> {out_data}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        default="<REPO_ROOT>/curve_plots",
    )
    parser.add_argument(
        "--balanced_dir",
        default="<CURVE_PLOTS_ROOT>",
    )
    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/separability_figures",
    )

    args = parser.parse_args()

    plot_variance_decomposition(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()