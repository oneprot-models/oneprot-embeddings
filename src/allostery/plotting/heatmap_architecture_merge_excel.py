#!/usr/bin/env python3
"""
export_auc_heatmaps_npz_with_excel_fill_to_excel.py

Exports plotting data for plot_auc_heatmaps_npz_with_excel_fill.py.

Creates:
    auc_heatmaps_npz_with_excel_fill_plotting_data.xlsx

Sheets:
    Heatmap_<embedding>
    Counts_<embedding>
    Sources_<embedding>
    AUC_long_final
    AUC_by_run_npz
    Excel_fallback_parsed
    Missing
    Config
    Model_key
    Dataset_key
    Embedding_key
    Task_key
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

DATASET_ORDER = ["PL8", "Kinase", "PL8 + Kinase", "ASD + PL8 + Kinase"]

DATASET_DISPLAY_NAMES = {
    "PL8": "PPI-Site",
    "Kinase": "KinSite",
    "PL8 + Kinase": "DualSite",
    "ASD + PL8 + Kinase": "AlloSite",
}

EXCEL_DATASET_ORDER = ["PPI-site", "Kinsite", "DualSite", "AlloSite"]

DATASET_TO_EXCEL = {
    "PL8": "PPI-site",
    "Kinase": "Kinsite",
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

EMBEDDING_ALIASES = {
    "Pockets": "p",
    "Pocket embeddings": "p",
    "Pocket": "p",
    "Pocket+text": "pt",
    "Pocket + Text": "pt",
    "Pocket and text embeddings": "pt",
    "Pocket+sequence": "ps",
    "Pocket + Sequence": "ps",
    "Pocket and sequence embeddings": "ps",
    "Pocket+sequence+text": "pst",
    "Pocket + Sequence + Text": "pst",
    "Pocket sequence and text embeddings": "pst",
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


def normalize_text(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def load_auc_from_npz(path):
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


def get_npz_path(base_dir, balanced_dir, dataset_name, embedding_key, model, seed):
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


def parse_excel_summary(excel_file):
    raw = pd.read_excel(excel_file, sheet_name=0, header=None)

    records = []
    dataset_names = set(EXCEL_DATASET_ORDER)
    model_names = set(MODELS)
    fallback_embedding_order = ["p", "ps", "pst", "pt"]

    current_dataset = None
    block_index = 0

    for i in range(len(raw)):
        first = normalize_text(raw.iloc[i, 0])

        if first in dataset_names:
            current_dataset = first
            block_index = 0
            continue

        if first != "Statistics by model_type:":
            continue

        if current_dataset is None:
            raise ValueError(f"Statistics block found before dataset near row {i + 1}")

        explicit_embedding = None

        for k in range(max(0, i - 6), i):
            candidate = normalize_text(raw.iloc[k, 0])
            if candidate in EMBEDDING_ALIASES:
                explicit_embedding = EMBEDDING_ALIASES[candidate]
                break

        if explicit_embedding is not None:
            embedding_key = explicit_embedding
        else:
            if block_index >= len(fallback_embedding_order):
                raise ValueError(
                    f"Could not infer embedding block for dataset {current_dataset}, "
                    f"block {block_index}, near row {i + 1}"
                )
            embedding_key = fallback_embedding_order[block_index]

        block_index += 1
        row = i + 1

        while row < len(raw):
            model_name = normalize_text(raw.iloc[row, 0])

            if model_name in dataset_names:
                break

            if model_name == "Statistics by model_type:":
                break

            if model_name in model_names:
                auc_value = raw.iloc[row, 1]
                std_value = raw.iloc[row, 2]
                n_value = raw.iloc[row, 3]

                records.append({
                    "excel_dataset": current_dataset,
                    "embedding_key": embedding_key,
                    "embedding_label": EMBEDDING_LABELS[embedding_key],
                    "model_id": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "excel_auc": float(auc_value),
                    "excel_std": float(std_value) if not pd.isna(std_value) else np.nan,
                    "excel_n": int(n_value) if not pd.isna(n_value) else 0,
                })

            row += 1

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No model results were parsed from the Excel file.")

    return df


def get_excel_value(excel_df, dataset_name, embedding_key, model):
    excel_dataset = DATASET_TO_EXCEL[dataset_name]

    hit = excel_df[
        (excel_df["excel_dataset"] == excel_dataset)
        & (excel_df["embedding_key"] == embedding_key)
        & (excel_df["model_id"] == model)
    ]

    if hit.empty:
        return None, None, None

    auc_value = float(hit["excel_auc"].iloc[0])
    std_value = float(hit["excel_std"].iloc[0]) if not pd.isna(hit["excel_std"].iloc[0]) else np.nan
    n_value = int(hit["excel_n"].iloc[0])

    return auc_value, std_value, n_value


def collect_all_data(base_dir, balanced_dir, excel_df):
    final_rows = []
    npz_run_rows = []
    missing_rows = []

    matrices = {}
    count_matrices = {}
    source_matrices = {}

    for embedding_key in EMBEDDING_ORDER:
        matrix = np.full((len(DATASET_ORDER), len(MODELS)), np.nan, dtype=float)
        count_matrix = np.zeros((len(DATASET_ORDER), len(MODELS)), dtype=int)
        source_matrix = np.full((len(DATASET_ORDER), len(MODELS)), "missing", dtype=object)

        for i, dataset_name in enumerate(DATASET_ORDER):
            for j, model in enumerate(MODELS):
                task = TASK_MAP[embedding_key][dataset_name]
                aucs = []

                for seed in SEEDS:
                    path = get_npz_path(
                        base_dir=base_dir,
                        balanced_dir=balanced_dir,
                        dataset_name=dataset_name,
                        embedding_key=embedding_key,
                        model=model,
                        seed=seed,
                    )

                    auc_value, error = load_auc_from_npz(path)

                    if auc_value is None:
                        missing_rows.append({
                            "dataset_original": dataset_name,
                            "dataset_display": DATASET_DISPLAY_NAMES[dataset_name],
                            "embedding_key": embedding_key,
                            "embedding_label": EMBEDDING_LABELS[embedding_key],
                            "task": task,
                            "model_id": model,
                            "model_label": MODEL_LABELS[model],
                            "seed": seed,
                            "source_file": path,
                            "reason": error,
                        })
                        continue

                    aucs.append(auc_value)

                    npz_run_rows.append({
                        "dataset_original": dataset_name,
                        "dataset_display": DATASET_DISPLAY_NAMES[dataset_name],
                        "embedding_key": embedding_key,
                        "embedding_label": EMBEDDING_LABELS[embedding_key],
                        "task": task,
                        "model_id": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "roc_auc": auc_value,
                        "source_file": path,
                    })

                if aucs:
                    final_auc = float(np.mean(aucs))
                    final_std = float(np.std(aucs))
                    final_n = len(aucs)
                    final_source = "npz"
                else:
                    excel_auc, excel_std, excel_n = get_excel_value(
                        excel_df=excel_df,
                        dataset_name=dataset_name,
                        embedding_key=embedding_key,
                        model=model,
                    )

                    if excel_auc is not None:
                        final_auc = excel_auc
                        final_std = excel_std
                        final_n = excel_n
                        final_source = "excel"
                    else:
                        final_auc = np.nan
                        final_std = np.nan
                        final_n = 0
                        final_source = "missing"

                matrix[i, j] = final_auc
                count_matrix[i, j] = final_n
                source_matrix[i, j] = final_source

                final_rows.append({
                    "dataset_original": dataset_name,
                    "dataset_display": DATASET_DISPLAY_NAMES[dataset_name],
                    "excel_dataset": DATASET_TO_EXCEL[dataset_name],
                    "embedding_key": embedding_key,
                    "embedding_label": EMBEDDING_LABELS[embedding_key],
                    "task": task,
                    "model_id": model,
                    "model_label": MODEL_LABELS[model],
                    "final_auc": final_auc,
                    "final_std": final_std,
                    "final_n": final_n,
                    "source_used": final_source,
                    "npz_n": len(aucs),
                    "npz_mean_auc": float(np.mean(aucs)) if aucs else np.nan,
                    "npz_std_auc": float(np.std(aucs)) if aucs else np.nan,
                    "excel_auc_used": final_auc if final_source == "excel" else np.nan,
                    "excel_std_used": final_std if final_source == "excel" else np.nan,
                    "excel_n_used": final_n if final_source == "excel" else np.nan,
                })

        matrices[embedding_key] = matrix
        count_matrices[embedding_key] = count_matrix
        source_matrices[embedding_key] = source_matrix

    return (
        pd.DataFrame(final_rows),
        pd.DataFrame(npz_run_rows),
        pd.DataFrame(missing_rows),
        matrices,
        count_matrices,
        source_matrices,
    )


def matrix_to_df(matrix):
    df = pd.DataFrame(
        matrix,
        index=[DATASET_DISPLAY_NAMES[d] for d in DATASET_ORDER],
        columns=[MODEL_LABELS[m] for m in MODELS],
    )
    return df.reset_index().rename(columns={"index": "Dataset"})


def autosize_excel_columns(writer, sheet_name, df, max_width=45):
    worksheet = writer.sheets[sheet_name]
    for col_idx, col_name in enumerate(df.columns, start=1):
        values = df[col_name].astype(str).head(500).tolist()
        max_len = max([len(str(col_name))] + [len(v) for v in values])
        width = min(max(max_len + 2, 10), max_width)
        col_letter = worksheet.cell(row=1, column=col_idx).column_letter
        worksheet.column_dimensions[col_letter].width = width
    worksheet.freeze_panes = "A2"


def export_excel(base_dir, balanced_dir, excel_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    excel_df = parse_excel_summary(excel_file)

    (
        final_df,
        npz_run_df,
        missing_df,
        matrices,
        count_matrices,
        source_matrices,
    ) = collect_all_data(base_dir, balanced_dir, excel_df)

    config_df = pd.DataFrame([
        {"parameter": "figure", "value": "AUC heatmaps NPZ with Excel fill"},
        {"parameter": "base_dir", "value": base_dir},
        {"parameter": "balanced_dir", "value": balanced_dir},
        {"parameter": "excel_file", "value": excel_file},
        {"parameter": "output_dir", "value": output_dir},
        {"parameter": "seeds", "value": ",".join(map(str, SEEDS))},
        {"parameter": "n_models", "value": len(MODELS)},
        {"parameter": "n_datasets", "value": len(DATASET_ORDER)},
        {"parameter": "n_embeddings", "value": len(EMBEDDING_ORDER)},
    ])

    model_key_df = pd.DataFrame([
        {"model_id": model, "model_label": MODEL_LABELS[model], "order": i + 1}
        for i, model in enumerate(MODELS)
    ])

    dataset_key_df = pd.DataFrame([
        {
            "dataset_original": dataset,
            "dataset_display": DATASET_DISPLAY_NAMES[dataset],
            "excel_dataset": DATASET_TO_EXCEL[dataset],
            "order": i + 1,
        }
        for i, dataset in enumerate(DATASET_ORDER)
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
            "dataset_original": dataset,
            "dataset_display": DATASET_DISPLAY_NAMES[dataset],
            "embedding_key": emb,
            "embedding_label": EMBEDDING_LABELS[emb],
            "task": TASK_MAP[emb][dataset],
        }
        for emb in EMBEDDING_ORDER
        for dataset in DATASET_ORDER
    ])

    out_xlsx = os.path.join(
        output_dir,
        "auc_heatmaps_npz_with_excel_fill_plotting_data.xlsx",
    )

    sheets_to_autosize = {}

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="AUC_long_final", index=False)
        npz_run_df.to_excel(writer, sheet_name="AUC_by_run_npz", index=False)
        excel_df.to_excel(writer, sheet_name="Excel_fallback_parsed", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)
        model_key_df.to_excel(writer, sheet_name="Model_key", index=False)
        dataset_key_df.to_excel(writer, sheet_name="Dataset_key", index=False)
        embedding_key_df.to_excel(writer, sheet_name="Embedding_key", index=False)
        task_key_df.to_excel(writer, sheet_name="Task_key", index=False)

        sheets_to_autosize.update({
            "AUC_long_final": final_df,
            "AUC_by_run_npz": npz_run_df,
            "Excel_fallback_parsed": excel_df,
            "Config": config_df,
            "Model_key": model_key_df,
            "Dataset_key": dataset_key_df,
            "Embedding_key": embedding_key_df,
            "Task_key": task_key_df,
        })

        if not missing_df.empty:
            missing_df.to_excel(writer, sheet_name="Missing", index=False)
            sheets_to_autosize["Missing"] = missing_df

        for emb in EMBEDDING_ORDER:
            label = emb.upper()

            heatmap_df = matrix_to_df(matrices[emb])
            counts_df = matrix_to_df(count_matrices[emb])
            sources_df = matrix_to_df(source_matrices[emb])

            heatmap_df.to_excel(writer, sheet_name=f"Heatmap_{label}", index=False)
            counts_df.to_excel(writer, sheet_name=f"Counts_{label}", index=False)
            sources_df.to_excel(writer, sheet_name=f"Sources_{label}", index=False)

            sheets_to_autosize[f"Heatmap_{label}"] = heatmap_df
            sheets_to_autosize[f"Counts_{label}"] = counts_df
            sheets_to_autosize[f"Sources_{label}"] = sources_df

        for sheet_name, df in sheets_to_autosize.items():
            autosize_excel_columns(writer, sheet_name, df)

    print(f"Saved Excel file -> {out_xlsx}")
    print("")
    print("Summary:")
    print(f"  Final heatmap cells : {len(final_df)}")
    print(f"  NPZ run-level rows  : {len(npz_run_df)}")
    print(f"  Excel fallback rows : {len(excel_df)}")
    print(f"  Missing NPZ rows    : {len(missing_df)}")


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
        "--excel_file",
        default="oneprot_allostery_results.xlsx",
    )

    parser.add_argument(
        "--output_dir",
        default="separability_figures",
    )

    args = parser.parse_args()

    export_excel(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        excel_file=args.excel_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()