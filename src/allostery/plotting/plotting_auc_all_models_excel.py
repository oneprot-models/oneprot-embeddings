#!/usr/bin/env python3
"""
export_figure6_roc_data_to_excel.py

Export ROC-curve plotting data for Figure 6 into an Excel workbook.

Outputs:
    figure6_architecture_roc_<embedding_key>_plotting_data.xlsx
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sklearn_auc


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
    "oneprot_pocket_text_32900": "#666666",
    "oneprot_struct_token_pocket_text_32900": "#0072B2",
    "oneprot_struct_graph_pocket_text_32900": "#009E73",
    "oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity": "#56B4E9",
    "oneprot_md_combined_gpcr_no_struct_graph_32900": "#E69F00",
    "oneprot_md_combined_gpcr_no_struct_token_32900": "#CC79A7",
    "oneprot_md_combined_gpcr_32900": "#D55E00",
}

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 400)

DATASET_ORDER = ["PPI-Site", "KinSite", "DualSite", "AlloDiverse"]

DATASET_META = {
    "PPI-Site": {"panel_label": "A", "regime": "Low separability", "source": "flat"},
    "KinSite": {"panel_label": "B", "regime": "Intermediate", "source": "flat"},
    "DualSite": {"panel_label": "C", "regime": "Intermediate-synergistic", "source": "flat"},
    "AlloDiverse": {"panel_label": "D", "regime": "High separability", "source": "balanced"},
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


def _load_npz(path):
    if not os.path.exists(path):
        return None

    data = np.load(path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = sklearn_auc(fpr, tpr)

    return fpr, tpr, roc_auc


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


def collect_model_data(load_fn, root_dir, dataset_name, model, task, embedding_key):
    seed_rows = []
    interp_tprs = []
    aucs = []

    for seed in SEEDS:
        result = load_fn(root_dir, model, task, seed)

        if result is None:
            continue

        fpr, tpr, roc_auc = result

        interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        interp_tprs.append(interp_tpr)
        aucs.append(roc_auc)

        for x, y in zip(fpr, tpr):
            seed_rows.append({
                "dataset": dataset_name,
                "embedding_key": embedding_key,
                "embedding_label": EMBEDDING_LABELS[embedding_key],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seed": seed,
                "fpr": x,
                "tpr": y,
                "seed_auc": roc_auc,
                "curve_type": "individual_seed",
            })

    if not interp_tprs:
        return None, seed_rows

    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)

    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    summary = {
        "dataset": dataset_name,
        "panel_label": DATASET_META[dataset_name]["panel_label"],
        "regime": DATASET_META[dataset_name]["regime"],
        "embedding_key": embedding_key,
        "embedding_label": EMBEDDING_LABELS[embedding_key],
        "model": model,
        "model_label": MODEL_LABELS[model],
        "color": MODEL_COLORS[model],
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "n_seeds": len(aucs),
        "seeds_used": ",".join(map(str, SEEDS)),
    }

    mean_rows = []
    for fpr_value, mean_value, std_value in zip(MEAN_FPR, mean_tpr, std_tpr):
        mean_rows.append({
            "dataset": dataset_name,
            "embedding_key": embedding_key,
            "embedding_label": EMBEDDING_LABELS[embedding_key],
            "model": model,
            "model_label": MODEL_LABELS[model],
            "color": MODEL_COLORS[model],
            "mean_fpr": fpr_value,
            "mean_tpr": mean_value,
            "std_tpr": std_value,
            "lower_tpr": max(mean_value - std_value, 0.0),
            "upper_tpr": min(mean_value + std_value, 1.0),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "n_seeds": len(aucs),
            "curve_type": "mean_interpolated",
        })

    return (summary, mean_rows), seed_rows


def export_excel(base_dir, balanced_dir, output_dir, embedding_key):
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    mean_curve_rows = []
    seed_curve_rows = []
    missing_rows = []

    for dataset_name in DATASET_ORDER:
        task = TASK_MAP[embedding_key][dataset_name]
        meta = DATASET_META[dataset_name]

        for model in MODELS:
            if meta["source"] == "balanced":
                result, seed_rows = collect_model_data(
                    load_balanced_hierarchical,
                    balanced_dir,
                    dataset_name,
                    model,
                    task,
                    embedding_key,
                )
            else:
                result, seed_rows = collect_model_data(
                    load_flat,
                    base_dir,
                    dataset_name,
                    model,
                    task,
                    embedding_key,
                )

            seed_curve_rows.extend(seed_rows)

            if result is None:
                missing_rows.append({
                    "dataset": dataset_name,
                    "embedding_key": embedding_key,
                    "task": task,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "reason": "No valid prediction files found",
                })
                continue

            summary, mean_rows = result
            summary_rows.append(summary)
            mean_curve_rows.extend(mean_rows)

    summary_df = pd.DataFrame(summary_rows)
    mean_curve_df = pd.DataFrame(mean_curve_rows)
    seed_curve_df = pd.DataFrame(seed_curve_rows)
    missing_df = pd.DataFrame(missing_rows)

    config_df = pd.DataFrame([
        {"parameter": "base_dir", "value": base_dir},
        {"parameter": "balanced_dir", "value": balanced_dir},
        {"parameter": "output_dir", "value": output_dir},
        {"parameter": "embedding_key", "value": embedding_key},
        {"parameter": "embedding_label", "value": EMBEDDING_LABELS[embedding_key]},
        {"parameter": "seeds", "value": ",".join(map(str, SEEDS))},
        {"parameter": "mean_fpr_points", "value": len(MEAN_FPR)},
    ])

    model_df = pd.DataFrame([
        {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "color": MODEL_COLORS[model],
        }
        for model in MODELS
    ])

    task_df = pd.DataFrame([
        {
            "embedding_key": emb,
            "embedding_label": EMBEDDING_LABELS[emb],
            "dataset": dataset,
            "task": task,
        }
        for emb, dataset_map in TASK_MAP.items()
        for dataset, task in dataset_map.items()
    ])

    out_xlsx = os.path.join(
        output_dir,
        f"figure6_architecture_roc_{embedding_key}_plotting_data.xlsx",
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="AUC_summary", index=False)
        mean_curve_df.to_excel(writer, sheet_name="ROC_curves_mean", index=False)
        seed_curve_df.to_excel(writer, sheet_name="ROC_curves_by_seed", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)
        model_df.to_excel(writer, sheet_name="Model_key", index=False)
        task_df.to_excel(writer, sheet_name="Task_key", index=False)

        if not missing_df.empty:
            missing_df.to_excel(writer, sheet_name="Missing", index=False)

        for sheet_name, df in {
            "AUC_summary": summary_df,
            "ROC_curves_mean": mean_curve_df,
            "ROC_curves_by_seed": seed_curve_df,
            "Config": config_df,
            "Model_key": model_df,
            "Task_key": task_df,
            "Missing": missing_df if not missing_df.empty else None,
        }.items():
            if df is None:
                continue

            worksheet = writer.sheets[sheet_name]
            for col_idx, col_name in enumerate(df.columns, start=1):
                max_len = max(
                    [len(str(col_name))]
                    + [len(str(v)) for v in df[col_name].head(200).values]
                )
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=col_idx).column_letter
                ].width = min(max_len + 2, 35)

            worksheet.freeze_panes = "A2"

    print(f"Saved Excel file -> {out_xlsx}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        default="<REPO_ROOT>/curve_plots",
        help="Directory with flat-format prediction .npz files.",
    )

    parser.add_argument(
        "--balanced_dir",
        default="<CURVE_PLOTS_ROOT>",
        help="Directory with hierarchical balanced AlloDiverse prediction files.",
    )

    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/figure6_model_comparison",
        help="Output directory.",
    )

    parser.add_argument(
        "--embedding_key",
        default="pst",
        choices=["p", "pt", "ps", "pst"],
        help="Fixed downstream embedding set.",
    )

    args = parser.parse_args()

    export_excel(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
        embedding_key=args.embedding_key,
    )


if __name__ == "__main__":
    main()