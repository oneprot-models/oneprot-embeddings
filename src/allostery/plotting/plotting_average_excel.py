#!/usr/bin/env python3
"""
export_average_roc_improved_data_to_excel.py

Exports plotting data for plot_average_roc_improved.py.

Creates:
    all_models_average_roc_plotting_data.xlsx

Sheets:
    AUC_summary
    ROC_curves_mean
    ROC_curves_by_seed
    Config
    Model_key
    Dataset_key
    Embedding_key
    Task_key
    Missing
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sklearn_auc


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

TASKS = [
    "ASD_pockets_binary_comp",
    "ASD_pockets_binary_text_comp",
    "ASD_pockets_sequence_binary_comp",
    "ASD_pockets_sequence_binary_text_comp",
    "Kinase_pocket",
    "Kinase_pocket_text",
    "Kinase_combined",
    "Kinase_combined_text",
    "merged_pocket_binary_comp",
    "merged_pocket_binary_text_comp",
    "merged_pocket_sequence_binary_comp",
    "merged_pocket_sequence_binary_text_comp",
    "ASD_merged_pocket_binary_comp",
    "ASD_merged_pocket_binary_text_comp",
    "ASD_merged_pocket_sequence_binary_comp",
    "ASD_merged_pocket_sequence_binary_text_comp",
]

DATASET_COLORS = {
    "PPI-Site": "#1f77b4",
    "KinSite": "#9467bd",
    "Dual-Site": "#2ca02c",
    "AlloDiverse": "#d62728",
}

EMBEDDING_LABELS = {
    "p": "Pocket",
    "pt": "Pocket+Text",
    "ps": "Pocket+Sequence",
    "pst": "Pocket+Sequence+Text",
}

EMBEDDING_STYLES = {
    "p": "-",
    "pt": "--",
    "ps": "-.",
    "pst": ":",
}

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 250)


def task_dataset(task):
    if task.startswith("ASD_merged"):
        return "AlloDiverse"
    if task.startswith("ASD_pockets"):
        return "PPI-Site"
    if task.startswith("merged"):
        return "Dual-Site"
    if task.startswith("Kinase"):
        return "KinSite"
    return "Other"


def task_embedding(task):
    if task == "Kinase_pocket":
        return "p"
    if task == "Kinase_pocket_text":
        return "pt"
    if task == "Kinase_combined":
        return "ps"
    if task == "Kinase_combined_text":
        return "pst"

    if task.endswith("_sequence_binary_text_comp"):
        return "pst"
    if task.endswith("_sequence_binary_comp"):
        return "ps"
    if task.endswith("_binary_text_comp"):
        return "pt"
    if task.endswith("_binary_comp"):
        return "p"

    raise ValueError(f"Could not determine embedding type for task: {task}")


def load_seed_curve(base_dir, model, task, seed):
    path = os.path.join(base_dir, f"{task}_{model}_seed{seed}_preds.npz")

    if not os.path.exists(path):
        return None, path, "Missing file"

    try:
        data = np.load(path)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
    except Exception as exc:
        return None, path, f"Unreadable file: {exc}"

    if len(np.unique(y_true)) < 2:
        return None, path, "Only one class present in y_true"

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = sklearn_auc(fpr, tpr)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "path": path,
    }, path, None


def collect_model_task_data(base_dir, model, task):
    dataset = task_dataset(task)
    emb = task_embedding(task)

    interpolated_tprs = []
    aucs = []
    seed_curve_rows = []
    missing_rows = []

    for seed in SEEDS:
        result, path, error = load_seed_curve(base_dir, model, task, seed)

        if result is None:
            missing_rows.append({
                "model": MODEL_LABELS[model],
                "model_id": model,
                "task": task,
                "dataset": dataset,
                "embedding_key": emb,
                "embedding": EMBEDDING_LABELS[emb],
                "seed": seed,
                "path": path,
                "reason": error,
            })
            continue

        fpr = result["fpr"]
        tpr = result["tpr"]
        roc_auc = result["auc"]

        interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        interpolated_tprs.append(interp_tpr)
        aucs.append(roc_auc)

        for point_idx, (x, y) in enumerate(zip(fpr, tpr)):
            seed_curve_rows.append({
                "model": MODEL_LABELS[model],
                "model_id": model,
                "task": task,
                "dataset": dataset,
                "dataset_color": DATASET_COLORS[dataset],
                "embedding_key": emb,
                "embedding": EMBEDDING_LABELS[emb],
                "embedding_linestyle": EMBEDDING_STYLES[emb],
                "seed": seed,
                "point_index": point_idx,
                "fpr": float(x),
                "tpr": float(y),
                "seed_auc": float(roc_auc),
                "source_file": result["path"],
            })

    if not interpolated_tprs:
        return None, [], seed_curve_rows, missing_rows

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    std_tpr = np.std(interpolated_tprs, axis=0)

    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    mean_curve_rows = []
    for point_idx, (x, y, sd) in enumerate(zip(MEAN_FPR, mean_tpr, std_tpr)):
        mean_curve_rows.append({
            "model": MODEL_LABELS[model],
            "model_id": model,
            "task": task,
            "dataset": dataset,
            "dataset_color": DATASET_COLORS[dataset],
            "embedding_key": emb,
            "embedding": EMBEDDING_LABELS[emb],
            "embedding_linestyle": EMBEDDING_STYLES[emb],
            "point_index": point_idx,
            "mean_fpr": float(x),
            "mean_tpr": float(y),
            "std_tpr": float(sd),
            "lower_tpr": float(max(y - sd, 0.0)),
            "upper_tpr": float(min(y + sd, 1.0)),
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "n_seeds": len(aucs),
            "seeds_used": ",".join(map(str, SEEDS)),
        })

    summary_row = {
        "model": MODEL_LABELS[model],
        "model_id": model,
        "task": task,
        "dataset": dataset,
        "dataset_color": DATASET_COLORS[dataset],
        "embedding_key": emb,
        "embedding": EMBEDDING_LABELS[emb],
        "embedding_linestyle": EMBEDDING_STYLES[emb],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "n_seeds": len(aucs),
        "seeds_used": ",".join(map(str, SEEDS)),
    }

    return summary_row, mean_curve_rows, seed_curve_rows, missing_rows


def autosize_excel_columns(writer, sheet_name, df, max_width=45):
    worksheet = writer.sheets[sheet_name]

    for col_idx, col_name in enumerate(df.columns, start=1):
        values = df[col_name].astype(str).head(500).tolist()
        max_len = max([len(str(col_name))] + [len(v) for v in values])
        width = min(max(max_len + 2, 10), max_width)
        col_letter = worksheet.cell(row=1, column=col_idx).column_letter
        worksheet.column_dimensions[col_letter].width = width

    worksheet.freeze_panes = "A2"


def export_excel(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    mean_curve_rows = []
    seed_curve_rows = []
    missing_rows = []

    for model in MODELS:
        for task in TASKS:
            summary, mean_rows, seed_rows, missing = collect_model_task_data(
                base_dir=base_dir,
                model=model,
                task=task,
            )

            if summary is not None:
                summary_rows.append(summary)

            mean_curve_rows.extend(mean_rows)
            seed_curve_rows.extend(seed_rows)
            missing_rows.extend(missing)

    summary_df = pd.DataFrame(summary_rows)
    mean_curve_df = pd.DataFrame(mean_curve_rows)
    seed_curve_df = pd.DataFrame(seed_curve_rows)
    missing_df = pd.DataFrame(missing_rows)

    config_df = pd.DataFrame([
        {"parameter": "figure", "value": "Supplementary average ROC curves"},
        {"parameter": "base_dir", "value": base_dir},
        {"parameter": "output_dir", "value": output_dir},
        {"parameter": "seeds", "value": ",".join(map(str, SEEDS))},
        {"parameter": "mean_fpr_points", "value": len(MEAN_FPR)},
        {"parameter": "n_models", "value": len(MODELS)},
        {"parameter": "n_tasks", "value": len(TASKS)},
    ])

    model_key_df = pd.DataFrame([
        {
            "model_id": model,
            "model": MODEL_LABELS[model],
        }
        for model in MODELS
    ])

    dataset_key_df = pd.DataFrame([
        {
            "dataset": dataset,
            "color": color,
        }
        for dataset, color in DATASET_COLORS.items()
    ])

    embedding_key_df = pd.DataFrame([
        {
            "embedding_key": emb,
            "embedding": EMBEDDING_LABELS[emb],
            "linestyle": EMBEDDING_STYLES[emb],
        }
        for emb in EMBEDDING_LABELS
    ])

    task_key_df = pd.DataFrame([
        {
            "task": task,
            "dataset": task_dataset(task),
            "embedding_key": task_embedding(task),
            "embedding": EMBEDDING_LABELS[task_embedding(task)],
        }
        for task in TASKS
    ])

    out_xlsx = os.path.join(
        output_dir,
        "all_models_average_roc_plotting_data.xlsx",
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="AUC_summary", index=False)
        mean_curve_df.to_excel(writer, sheet_name="ROC_curves_mean", index=False)
        seed_curve_df.to_excel(writer, sheet_name="ROC_curves_by_seed", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)
        model_key_df.to_excel(writer, sheet_name="Model_key", index=False)
        dataset_key_df.to_excel(writer, sheet_name="Dataset_key", index=False)
        embedding_key_df.to_excel(writer, sheet_name="Embedding_key", index=False)
        task_key_df.to_excel(writer, sheet_name="Task_key", index=False)

        if not missing_df.empty:
            missing_df.to_excel(writer, sheet_name="Missing", index=False)

        sheets = {
            "AUC_summary": summary_df,
            "ROC_curves_mean": mean_curve_df,
            "ROC_curves_by_seed": seed_curve_df,
            "Config": config_df,
            "Model_key": model_key_df,
            "Dataset_key": dataset_key_df,
            "Embedding_key": embedding_key_df,
            "Task_key": task_key_df,
        }

        if not missing_df.empty:
            sheets["Missing"] = missing_df

        for sheet_name, df in sheets.items():
            autosize_excel_columns(writer, sheet_name, df)

    print(f"Saved Excel file -> {out_xlsx}")
    print("")
    print("Summary:")
    print(f"  AUC rows       : {len(summary_df)}")
    print(f"  Mean ROC rows  : {len(mean_curve_df)}")
    print(f"  Seed ROC rows  : {len(seed_curve_df)}")
    print(f"  Missing rows   : {len(missing_df)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/curve_plots",
        help="Directory containing flat-format prediction .npz files.",
    )

    parser.add_argument(
        "--output_dir",
        default="/p/project1/hai_oneprot/bazarova1/oneprot-panda/average_roc_plots",
        help="Directory where the Excel workbook will be written.",
    )

    args = parser.parse_args()

    export_excel(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()