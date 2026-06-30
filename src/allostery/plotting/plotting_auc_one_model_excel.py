#!/usr/bin/env python3
"""
export_figure2_representative_roc_data_to_excel.py

Exports the plotting data for plot_figure2_representative_roc_clean.py.

Creates an Excel workbook containing:
    1. AUC_summary
       - mean ROC-AUC, SD ROC-AUC, number of seeds per dataset/embedding

    2. ROC_curves_mean
       - interpolated mean ROC curve used for plotting
       - mean FPR, mean TPR, SD TPR, lower/upper shaded band

    3. ROC_curves_by_seed
       - original ROC curve points for each individual seed

    4. Config
       - paths, model, seeds, embedding definitions

    5. Missing
       - any missing or invalid prediction files

Usage:
    python export_figure2_representative_roc_data_to_excel.py \
        --base_dir <REPO_ROOT>/curve_plots \
        --balanced_dir <CURVE_PLOTS_ROOT> \
        --output_dir <REPO_ROOT>/figure2_representative
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sklearn_auc


MODEL = "oneprot_md_combined_gpcr_no_struct_graph_32900"
MODEL_LABEL = "MD+ST+Pocket+Text"

SEEDS = [11, 12, 13, 14, 15]
MEAN_FPR = np.linspace(0, 1, 300)

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

DATASETS = [
    {
        "panel_label": "A",
        "panel_title": "PPI-Site",
        "regime": "Low separability",
        "source": "flat",
        "tasks": {
            "p": "ASD_pockets_binary_comp",
            "pt": "ASD_pockets_binary_text_comp",
            "ps": "ASD_pockets_sequence_binary_comp",
            "pst": "ASD_pockets_sequence_binary_text_comp",
        },
    },
    {
        "panel_label": "B",
        "panel_title": "KinSite",
        "regime": "Intermediate",
        "source": "flat",
        "tasks": {
            "p": "Kinase_pocket",
            "pt": "Kinase_pocket_text",
            "ps": "Kinase_combined",
            "pst": "Kinase_combined_text",
        },
    },
    {
        "panel_label": "C",
        "panel_title": "Dual-Site",
        "regime": "Intermediate-synergistic",
        "source": "flat",
        "tasks": {
            "p": "merged_pocket_binary_comp",
            "pt": "merged_pocket_binary_text_comp",
            "ps": "merged_pocket_sequence_binary_comp",
            "pst": "merged_pocket_sequence_binary_text_comp",
        },
    },
    {
        "panel_label": "D",
        "panel_title": "AlloDiverse",
        "regime": "High separability",
        "source": "balanced",
        "tasks": {
            "p": "ASD_merged_pocket_binary_comp",
            "pt": "ASD_merged_pocket_binary_text_comp",
            "ps": "ASD_merged_pocket_sequence_binary_comp",
            "pst": "ASD_merged_pocket_sequence_binary_text_comp",
        },
    },
]


def _load_npz(path):
    if not os.path.exists(path):
        return None

    try:
        data = np.load(path)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
    except Exception:
        return None

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = sklearn_auc(fpr, tpr)

    return {
        "path": path,
        "y_true": y_true,
        "y_pred": y_pred,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
    }


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


def collect_dataset_embedding_data(dataset_cfg, emb_key, base_dir, balanced_dir):
    task = dataset_cfg["tasks"][emb_key]

    if dataset_cfg["source"] == "flat":
        load_fn = load_flat
        root_dir = base_dir
    else:
        load_fn = load_balanced_hierarchical
        root_dir = balanced_dir

    aucs = []
    interpolated_tprs = []
    seed_curve_rows = []
    raw_prediction_rows = []
    missing_rows = []

    for seed in SEEDS:
        result = load_fn(root_dir, MODEL, task, seed)

        if result is None:
            missing_rows.append({
                "dataset": dataset_cfg["panel_title"],
                "panel_label": dataset_cfg["panel_label"],
                "regime": dataset_cfg["regime"],
                "embedding_key": emb_key,
                "embedding_label": EMBEDDING_LABELS[emb_key],
                "task": task,
                "model": MODEL,
                "model_label": MODEL_LABEL,
                "seed": seed,
                "source": dataset_cfg["source"],
                "reason": "Missing file, unreadable file, or y_true has only one class",
            })
            continue

        fpr = result["fpr"]
        tpr = result["tpr"]
        roc_auc = result["auc"]

        interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        aucs.append(roc_auc)
        interpolated_tprs.append(interp_tpr)

        for point_idx, (x, y) in enumerate(zip(fpr, tpr)):
            seed_curve_rows.append({
                "dataset": dataset_cfg["panel_title"],
                "panel_label": dataset_cfg["panel_label"],
                "regime": dataset_cfg["regime"],
                "embedding_key": emb_key,
                "embedding_label": EMBEDDING_LABELS[emb_key],
                "embedding_color": EMBEDDING_COLORS[emb_key],
                "task": task,
                "model": MODEL,
                "model_label": MODEL_LABEL,
                "seed": seed,
                "point_index": point_idx,
                "fpr": float(x),
                "tpr": float(y),
                "seed_auc": float(roc_auc),
                "source_file": result["path"],
            })

        for obs_idx, (yt, yp) in enumerate(zip(result["y_true"], result["y_pred"])):
            raw_prediction_rows.append({
                "dataset": dataset_cfg["panel_title"],
                "embedding_key": emb_key,
                "embedding_label": EMBEDDING_LABELS[emb_key],
                "task": task,
                "model": MODEL,
                "model_label": MODEL_LABEL,
                "seed": seed,
                "observation_index": obs_idx,
                "y_true": int(yt),
                "y_pred": float(yp),
                "source_file": result["path"],
            })

    if not interpolated_tprs:
        return None, [], seed_curve_rows, raw_prediction_rows, missing_rows

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    std_tpr = np.std(interpolated_tprs, axis=0)

    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    mean_curve_rows = []
    for point_idx, (x, y, sd) in enumerate(zip(MEAN_FPR, mean_tpr, std_tpr)):
        mean_curve_rows.append({
            "dataset": dataset_cfg["panel_title"],
            "panel_label": dataset_cfg["panel_label"],
            "regime": dataset_cfg["regime"],
            "embedding_key": emb_key,
            "embedding_label": EMBEDDING_LABELS[emb_key],
            "embedding_color": EMBEDDING_COLORS[emb_key],
            "task": task,
            "model": MODEL,
            "model_label": MODEL_LABEL,
            "point_index": point_idx,
            "mean_fpr": float(x),
            "mean_tpr": float(y),
            "std_tpr": float(sd),
            "lower_tpr": float(max(y - sd, 0.0)),
            "upper_tpr": float(min(y + sd, 1.0)),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "n_seeds": len(aucs),
            "seeds_used": ",".join(map(str, SEEDS)),
        })

    summary_row = {
        "dataset": dataset_cfg["panel_title"],
        "panel_label": dataset_cfg["panel_label"],
        "regime": dataset_cfg["regime"],
        "embedding_key": emb_key,
        "embedding_label": EMBEDDING_LABELS[emb_key],
        "embedding_color": EMBEDDING_COLORS[emb_key],
        "task": task,
        "model": MODEL,
        "model_label": MODEL_LABEL,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "n_seeds": len(aucs),
        "seeds_used": ",".join(map(str, SEEDS)),
    }

    return summary_row, mean_curve_rows, seed_curve_rows, raw_prediction_rows, missing_rows


def autosize_excel_columns(writer, sheet_name, df, max_width=45):
    worksheet = writer.sheets[sheet_name]

    for col_idx, col_name in enumerate(df.columns, start=1):
        values = df[col_name].astype(str).head(500).tolist()
        max_len = max([len(str(col_name))] + [len(v) for v in values])
        width = min(max(max_len + 2, 10), max_width)
        col_letter = worksheet.cell(row=1, column=col_idx).column_letter
        worksheet.column_dimensions[col_letter].width = width

    worksheet.freeze_panes = "A2"


def export_excel(base_dir, balanced_dir, output_dir, include_raw_predictions):
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    mean_curve_rows = []
    seed_curve_rows = []
    raw_prediction_rows = []
    missing_rows = []

    for dataset_cfg in DATASETS:
        for emb_key in EMBEDDING_ORDER:
            (
                summary_row,
                mean_rows,
                seed_rows,
                raw_rows,
                missing,
            ) = collect_dataset_embedding_data(
                dataset_cfg=dataset_cfg,
                emb_key=emb_key,
                base_dir=base_dir,
                balanced_dir=balanced_dir,
            )

            if summary_row is not None:
                summary_rows.append(summary_row)

            mean_curve_rows.extend(mean_rows)
            seed_curve_rows.extend(seed_rows)
            missing_rows.extend(missing)

            if include_raw_predictions:
                raw_prediction_rows.extend(raw_rows)

    summary_df = pd.DataFrame(summary_rows)
    mean_curve_df = pd.DataFrame(mean_curve_rows)
    seed_curve_df = pd.DataFrame(seed_curve_rows)
    missing_df = pd.DataFrame(missing_rows)

    config_df = pd.DataFrame([
        {"parameter": "figure", "value": "Figure 2 representative ROC curves"},
        {"parameter": "model", "value": MODEL},
        {"parameter": "model_label", "value": MODEL_LABEL},
        {"parameter": "base_dir", "value": base_dir},
        {"parameter": "balanced_dir", "value": balanced_dir},
        {"parameter": "output_dir", "value": output_dir},
        {"parameter": "seeds", "value": ",".join(map(str, SEEDS))},
        {"parameter": "mean_fpr_points", "value": len(MEAN_FPR)},
        {"parameter": "include_raw_predictions", "value": include_raw_predictions},
    ])

    embedding_key_df = pd.DataFrame([
        {
            "embedding_key": emb_key,
            "embedding_label": EMBEDDING_LABELS[emb_key],
            "embedding_color": EMBEDDING_COLORS[emb_key],
        }
        for emb_key in EMBEDDING_ORDER
    ])

    task_key_df = pd.DataFrame([
        {
            "dataset": dataset_cfg["panel_title"],
            "panel_label": dataset_cfg["panel_label"],
            "regime": dataset_cfg["regime"],
            "source": dataset_cfg["source"],
            "embedding_key": emb_key,
            "embedding_label": EMBEDDING_LABELS[emb_key],
            "task": dataset_cfg["tasks"][emb_key],
        }
        for dataset_cfg in DATASETS
        for emb_key in EMBEDDING_ORDER
    ])

    out_xlsx = os.path.join(
        output_dir,
        "figure2_representative_roc_plotting_data.xlsx",
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="AUC_summary", index=False)
        mean_curve_df.to_excel(writer, sheet_name="ROC_curves_mean", index=False)
        seed_curve_df.to_excel(writer, sheet_name="ROC_curves_by_seed", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)
        embedding_key_df.to_excel(writer, sheet_name="Embedding_key", index=False)
        task_key_df.to_excel(writer, sheet_name="Task_key", index=False)

        if not missing_df.empty:
            missing_df.to_excel(writer, sheet_name="Missing", index=False)

        if include_raw_predictions:
            raw_prediction_df = pd.DataFrame(raw_prediction_rows)
            raw_prediction_df.to_excel(writer, sheet_name="Raw_predictions", index=False)

        sheets = {
            "AUC_summary": summary_df,
            "ROC_curves_mean": mean_curve_df,
            "ROC_curves_by_seed": seed_curve_df,
            "Config": config_df,
            "Embedding_key": embedding_key_df,
            "Task_key": task_key_df,
        }

        if not missing_df.empty:
            sheets["Missing"] = missing_df

        if include_raw_predictions:
            sheets["Raw_predictions"] = raw_prediction_df

        for sheet_name, df in sheets.items():
            autosize_excel_columns(writer, sheet_name, df)

    print(f"Saved Excel file -> {out_xlsx}")

    print("\nSummary:")
    print(f"  AUC rows            : {len(summary_df)}")
    print(f"  Mean ROC rows       : {len(mean_curve_df)}")
    print(f"  Seed ROC rows       : {len(seed_curve_df)}")
    print(f"  Missing entries     : {len(missing_df)}")

    if include_raw_predictions:
        print(f"  Raw prediction rows : {len(raw_prediction_rows)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        default="<REPO_ROOT>/curve_plots",
        help="Directory containing flat-format prediction .npz files.",
    )

    parser.add_argument(
        "--balanced_dir",
        default="<CURVE_PLOTS_ROOT>",
        help="Directory containing balanced hierarchical AlloDiverse prediction files.",
    )

    parser.add_argument(
        "--output_dir",
        default="<REPO_ROOT>/figure2_representative",
        help="Directory where the Excel workbook will be written.",
    )

    parser.add_argument(
        "--include_raw_predictions",
        action="store_true",
        help=(
            "Also export y_true and y_pred for every seed/sample. "
            "This may create a large Excel file."
        ),
    )

    args = parser.parse_args()

    export_excel(
        base_dir=args.base_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
        include_raw_predictions=args.include_raw_predictions,
    )


if __name__ == "__main__":
    main()