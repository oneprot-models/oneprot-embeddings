#!/usr/bin/env python3
"""
export_auc_heatmap_dataset_by_embedding_to_excel.py

Exports the plotting data for plot_auc_heatmap_dataset_by_embedding.py.

Creates:
    auc_heatmap_dataset_by_embedding_plotting_data.xlsx

Sheets:
    Heatmap_matrix
    Heatmap_counts
    AUC_long
    AUC_by_run
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

SEEDS = [11, 12, 13, 14, 15]

DATASET_ORDER = [
    "PL8",
    "Kinase",
    "PL8 + Kinase",
    "ASD + PL8 + Kinase",
]

DATASET_DISPLAY_NAMES = {
    "PL8": "PPI-Site",
    "Kinase": "KinSite",
    "PL8 + Kinase": "DualSite",
    "ASD + PL8 + Kinase": "AlloSite",
}

EMBEDDING_ORDER = ["p", "pt", "ps", "pst"]

EMBEDDING_LABELS = {
    "p": "Pocket",
    "pt": "Pocket + Text",
    "ps": "Pocket + Sequence",
    "pst": "Pocket + Sequence + Text",
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
        return None, "Missing file"

    try:
        data = np.load(path)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
    except Exception as exc:
        return None, f"Unreadable file: {exc}"

    if len(np.unique(y_true)) < 2:
        return None, "Only one class present in y_true"

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(sklearn_auc(fpr, tpr)), None


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


def collect_data(base_dir, balanced_dir):
    auc_by_run_rows = []
    auc_long_rows = []
    missing_rows = []

    for dataset_name in DATASET_ORDER:
        for embedding_key in EMBEDDING_ORDER:
            task = TASK_MAP[embedding_key][dataset_name]
            aucs = []

            for model in MODELS:
                for seed in SEEDS:
                    path = get_path(
                        base_dir=base_dir,
                        balanced_dir=balanced_dir,
                        dataset_name=dataset_name,
                        embedding_key=embedding_key,
                        model=model,
                        seed=seed,
                    )

                    auc_value, error = load_auc(path)

                    row_base = {
                        "dataset_original": dataset_name,
                        "dataset_display": DATASET_DISPLAY_NAMES[dataset_name],
                        "embedding_key": embedding_key,
                        "embedding_label": EMBEDDING_LABELS[embedding_key],
                        "task": task,
                        "model_id": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "seed": seed,
                        "source_file": path,
                    }

                    if auc_value is None:
                        missing_rows.append({**row_base, "reason": error})
                        continue

                    aucs.append(auc_value)

                    auc_by_run_rows.append({
                        **row_base,
                        "roc_auc": auc_value,
                    })

            if aucs:
                auc_long_rows.append({
                    "dataset_original": dataset_name,
                    "dataset_display": DATASET_DISPLAY_NAMES[dataset_name],
                    "embedding_key": embedding_key,
                    "embedding_label": EMBEDDING_LABELS[embedding_key],
                    "task": task,
                    "mean_auc": float(np.mean(aucs)),
                    "std_auc": float(np.std(aucs)),
                    "min_auc": float(np.min(aucs)),
                    "max_auc": float(np.max(aucs)),
                    "n_runs": len(aucs),
                    "n_models": len(MODELS),
                    "n_seeds": len(SEEDS),
                    "seeds_used": ",".join(map(str, SEEDS)),
                })
            else:
                auc_long_rows.append({
                    "dataset_original": dataset_name,
                    "dataset_display": DATASET_DISPLAY_NAMES[dataset_name],
                    "embedding_key": embedding_key,
                    "embedding_label": EMBEDDING_LABELS[embedding_key],
                    "task": task,
                    "mean_auc": np.nan,
                    "std_auc": np.nan,
                    "min_auc": np.nan,
                    "max_auc": np.nan,
                    "n_runs": 0,
                    "n_models": len(MODELS),
                    "n_seeds": len(SEEDS),
                    "seeds_used": ",".join(map(str, SEEDS)),
                })

    return (
        pd.DataFrame(auc_long_rows),
        pd.DataFrame(auc_by_run_rows),
        pd.DataFrame(missing_rows),
    )


def make_heatmap_matrix(auc_long_df, value_col):
    matrix = auc_long_df.pivot(
        index="dataset_display",
        columns="embedding_label",
        values=value_col,
    )

    matrix = matrix.reindex(
        index=[DATASET_DISPLAY_NAMES[d] for d in DATASET_ORDER],
        columns=[EMBEDDING_LABELS[e] for e in EMBEDDING_ORDER],
    )

    return matrix.reset_index().rename(columns={"dataset_display": "Dataset"})


def autosize_excel_columns(writer, sheet_name, df, max_width=45):
    worksheet = writer.sheets[sheet_name]

    for col_idx, col_name in enumerate(df.columns, start=1):
        values = df[col_name].astype(str).head(500).tolist()
        max_len = max([len(str(col_name))] + [len(v) for v in values])
        width = min(max(max_len + 2, 10), max_width)
        col_letter = worksheet.cell(row=1, column=col_idx).column_letter
        worksheet.column_dimensions[col_letter].width = width

    worksheet.freeze_panes = "A2"


def export_excel(base_dir, balanced_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    auc_long_df, auc_by_run_df, missing_df = collect_data(base_dir, balanced_dir)

    heatmap_matrix_df = make_heatmap_matrix(auc_long_df, "mean_auc")
    heatmap_counts_df = make_heatmap_matrix(auc_long_df, "n_runs")

    config_df = pd.DataFrame([
        {"parameter": "figure", "value": "Mean ROC-AUC heatmap by dataset and embedding"},
        {"parameter": "base_dir", "value": base_dir},
        {"parameter": "balanced_dir", "value": balanced_dir},
        {"parameter": "output_dir", "value": output_dir},
        {"parameter": "seeds", "value": ",".join(map(str, SEEDS))},
        {"parameter": "n_models", "value": len(MODELS)},
        {"parameter": "n_datasets", "value": len(DATASET_ORDER)},
        {"parameter": "n_embeddings", "value": len(EMBEDDING_ORDER)},
    ])

    model_key_df = pd.DataFrame([
        {
            "model_id": model,
            "model_label": MODEL_LABELS.get(model, model),
        }
        for model in MODELS
    ])

    dataset_key_df = pd.DataFrame([
        {
            "dataset_original": dataset_name,
            "dataset_display": DATASET_DISPLAY_NAMES[dataset_name],
            "order": i + 1,
            "source": "balanced_hierarchical" if dataset_name == "ASD + PL8 + Kinase" else "flat",
        }
        for i, dataset_name in enumerate(DATASET_ORDER)
    ])

    embedding_key_df = pd.DataFrame([
        {
            "embedding_key": emb,
            "embedding_label": EMBEDDING_LABELS[emb],
            "order": i + 1,
        }
        for i, emb in enumerate(EMBEDDING_ORDER)
    ])

    task_key_df = pd.DataFrame([
        {
            "dataset_original": dataset_name,
            "dataset_display": DATASET_DISPLAY_NAMES[dataset_name],
            "embedding_key": emb,
            "embedding_label": EMBEDDING_LABELS[emb],
            "task": TASK_MAP[emb][dataset_name],
        }
        for dataset_name in DATASET_ORDER
        for emb in EMBEDDING_ORDER
    ])

    out_xlsx = os.path.join(
        output_dir,
        "auc_heatmap_dataset_by_embedding_plotting_data.xlsx",
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        heatmap_matrix_df.to_excel(writer, sheet_name="Heatmap_matrix", index=False)
        heatmap_counts_df.to_excel(writer, sheet_name="Heatmap_counts", index=False)
        auc_long_df.to_excel(writer, sheet_name="AUC_long", index=False)
        auc_by_run_df.to_excel(writer, sheet_name="AUC_by_run", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)
        model_key_df.to_excel(writer, sheet_name="Model_key", index=False)
        dataset_key_df.to_excel(writer, sheet_name="Dataset_key", index=False)
        embedding_key_df.to_excel(writer, sheet_name="Embedding_key", index=False)
        task_key_df.to_excel(writer, sheet_name="Task_key", index=False)

        if not missing_df.empty:
            missing_df.to_excel(writer, sheet_name="Missing", index=False)

        sheets = {
            "Heatmap_matrix": heatmap_matrix_df,
            "Heatmap_counts": heatmap_counts_df,
            "AUC_long": auc_long_df,
            "AUC_by_run": auc_by_run_df,
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
    print(f"  Heatmap cells : {len(auc_long_df)}")
    print(f"  Run-level AUC : {len(auc_by_run_df)}")
    print(f"  Missing rows  : {len(missing_df)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        default="curve_plots",
    )

    parser.add_argument(
        "--balanced_dir",
        default="curve_plots_balanced",
    )

    parser.add_argument(
        "--output_dir",
        default="separability_figures",
    )

    args = parser.parse_args()

    export_excel(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()