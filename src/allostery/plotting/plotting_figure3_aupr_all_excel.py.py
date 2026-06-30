#!/usr/bin/env python3
"""
export_supplement_all_models_pr_data_to_excel.py

Exports plotting data for plot_supplement_all_models_pr_clean.py.

Creates:
    supplement_all_models_pr_plotting_data.xlsx

Sheets:
    AUPR_summary
    PR_curves_mean
    PR_curves_by_seed
    Config
    Model_key
    Embedding_key
    Condition_key
    Task_key
    Missing
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc


DEFAULT_UNBALANCED_DIR = "<REPO_ROOT>/curve_plots"
DEFAULT_BALANCED_DIR = "<CURVE_PLOTS_ROOT>"
DEFAULT_OUTPUT_DIR = "<REPO_ROOT>/figure3_imbalance_pr"


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

CONDITIONS = [
    {
        "condition_key": "allodiverse_original",
        "condition_label": "AlloDiverse, original training",
        "dataset": "AlloDiverse",
        "linestyle": "-",
        "loader": "flat",
        "tasks": ALLODIVERSE_TASKS,
    },
    {
        "condition_key": "allodiverse_balanced",
        "condition_label": "AlloDiverse, balanced training",
        "dataset": "AlloDiverse",
        "linestyle": "--",
        "loader": "balanced",
        "tasks": ALLODIVERSE_TASKS,
    },
    {
        "condition_key": "kinsite",
        "condition_label": "KinSite",
        "dataset": "KinSite",
        "linestyle": ":",
        "loader": "flat",
        "tasks": KINSITE_TASKS,
    },
]


def extract_scores(y_pred):
    arr = np.asarray(y_pred)

    if arr.ndim == 1:
        return arr.astype(float)

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        return arr[:, -1].astype(float)

    raise ValueError(f"Unsupported y_pred shape: {arr.shape}")


def _load_npz(path):
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

    return (y_true, y_pred), None


def load_flat(root_dir, model, task, seed):
    path = os.path.join(root_dir, f"{task}_{model}_seed{seed}_preds.npz")
    loaded, error = _load_npz(path)
    return loaded, path, error


def load_balanced_hierarchical(root_dir, model, task, seed):
    path = os.path.join(
        root_dir,
        model,
        f"{task}_balanced",
        f"seed{seed}",
        "preds.npz",
    )
    loaded, error = _load_npz(path)
    return loaded, path, error


def compute_pr_curve(y_true, y_pred):
    scores = extract_scores(y_pred)

    precision, recall, _ = precision_recall_curve(y_true, scores)

    recall = recall[::-1]
    precision = precision[::-1]

    aupr = sklearn_auc(recall, precision)
    prevalence = float(np.mean(y_true))

    return recall, precision, aupr, prevalence


def collect_pr_data(
    unbalanced_dir,
    balanced_dir,
    model_id,
    model_label,
    emb_key,
    condition,
):
    task = condition["tasks"][emb_key]

    if condition["loader"] == "flat":
        load_fn = load_flat
        root_dir = unbalanced_dir
    elif condition["loader"] == "balanced":
        load_fn = load_balanced_hierarchical
        root_dir = balanced_dir
    else:
        raise ValueError(f"Unknown loader: {condition['loader']}")

    precisions = []
    auprs = []
    prevalences = []

    seed_curve_rows = []
    missing_rows = []

    for seed in SEEDS:
        loaded, path, error = load_fn(root_dir, model_id, task, seed)

        if loaded is None:
            missing_rows.append({
                "model": model_label,
                "model_id": model_id,
                "embedding_key": emb_key,
                "embedding_label": EMBEDDING_LABELS[emb_key],
                "condition_key": condition["condition_key"],
                "condition_label": condition["condition_label"],
                "dataset": condition["dataset"],
                "task": task,
                "seed": seed,
                "path": path,
                "reason": error,
            })
            continue

        y_true, y_pred = loaded
        recall, precision, aupr, prevalence = compute_pr_curve(y_true, y_pred)

        interp_precision = np.interp(MEAN_RECALL, recall, precision)

        precisions.append(interp_precision)
        auprs.append(aupr)
        prevalences.append(prevalence)

        for point_idx, (r, p) in enumerate(zip(recall, precision)):
            seed_curve_rows.append({
                "model": model_label,
                "model_id": model_id,
                "embedding_key": emb_key,
                "embedding_label": EMBEDDING_LABELS[emb_key],
                "embedding_color": EMBEDDING_COLORS[emb_key],
                "condition_key": condition["condition_key"],
                "condition_label": condition["condition_label"],
                "dataset": condition["dataset"],
                "linestyle": condition["linestyle"],
                "task": task,
                "seed": seed,
                "point_index": point_idx,
                "recall": float(r),
                "precision": float(p),
                "seed_aupr": float(aupr),
                "seed_prevalence": float(prevalence),
                "source_file": path,
            })

    if not precisions:
        return None, [], seed_curve_rows, missing_rows

    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)

    mean_aupr = float(np.mean(auprs))
    std_aupr = float(np.std(auprs))
    mean_prevalence = float(np.mean(prevalences))
    std_prevalence = float(np.std(prevalences))

    mean_curve_rows = []
    for point_idx, (r, p, sd) in enumerate(zip(MEAN_RECALL, mean_precision, std_precision)):
        mean_curve_rows.append({
            "model": model_label,
            "model_id": model_id,
            "embedding_key": emb_key,
            "embedding_label": EMBEDDING_LABELS[emb_key],
            "embedding_color": EMBEDDING_COLORS[emb_key],
            "condition_key": condition["condition_key"],
            "condition_label": condition["condition_label"],
            "dataset": condition["dataset"],
            "linestyle": condition["linestyle"],
            "task": task,
            "point_index": point_idx,
            "mean_recall": float(r),
            "mean_precision": float(p),
            "std_precision": float(sd),
            "lower_precision": float(max(p - sd, 0.0)),
            "upper_precision": float(min(p + sd, 1.0)),
            "mean_aupr": mean_aupr,
            "std_aupr": std_aupr,
            "mean_prevalence": mean_prevalence,
            "std_prevalence": std_prevalence,
            "n_seeds": len(auprs),
            "seeds_used": ",".join(map(str, SEEDS)),
        })

    summary_row = {
        "model": model_label,
        "model_id": model_id,
        "embedding_key": emb_key,
        "embedding_label": EMBEDDING_LABELS[emb_key],
        "embedding_color": EMBEDDING_COLORS[emb_key],
        "condition_key": condition["condition_key"],
        "condition_label": condition["condition_label"],
        "dataset": condition["dataset"],
        "linestyle": condition["linestyle"],
        "task": task,
        "mean_aupr": mean_aupr,
        "std_aupr": std_aupr,
        "mean_prevalence": mean_prevalence,
        "std_prevalence": std_prevalence,
        "n_seeds": len(auprs),
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


def export_excel(unbalanced_dir, balanced_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    mean_curve_rows = []
    seed_curve_rows = []
    missing_rows = []

    for model_id, model_label in MODELS:
        for emb_key in EMBEDDING_ORDER:
            for condition in CONDITIONS:
                summary, mean_rows, seed_rows, missing = collect_pr_data(
                    unbalanced_dir=unbalanced_dir,
                    balanced_dir=balanced_dir,
                    model_id=model_id,
                    model_label=model_label,
                    emb_key=emb_key,
                    condition=condition,
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
        {"parameter": "figure", "value": "Supplementary all-model PR curves"},
        {"parameter": "unbalanced_dir", "value": unbalanced_dir},
        {"parameter": "balanced_dir", "value": balanced_dir},
        {"parameter": "output_dir", "value": output_dir},
        {"parameter": "seeds", "value": ",".join(map(str, SEEDS))},
        {"parameter": "mean_recall_points", "value": len(MEAN_RECALL)},
        {"parameter": "n_models", "value": len(MODELS)},
        {"parameter": "n_embeddings", "value": len(EMBEDDING_ORDER)},
        {"parameter": "n_conditions", "value": len(CONDITIONS)},
    ])

    model_key_df = pd.DataFrame([
        {"model_id": model_id, "model": model_label}
        for model_id, model_label in MODELS
    ])

    embedding_key_df = pd.DataFrame([
        {
            "embedding_key": emb,
            "embedding_label": EMBEDDING_LABELS[emb],
            "color": EMBEDDING_COLORS[emb],
        }
        for emb in EMBEDDING_ORDER
    ])

    condition_key_df = pd.DataFrame([
        {
            "condition_key": c["condition_key"],
            "condition_label": c["condition_label"],
            "dataset": c["dataset"],
            "linestyle": c["linestyle"],
            "loader": c["loader"],
        }
        for c in CONDITIONS
    ])

    task_key_rows = []
    for emb in EMBEDDING_ORDER:
        task_key_rows.append({
            "embedding_key": emb,
            "embedding_label": EMBEDDING_LABELS[emb],
            "condition_key": "allodiverse_original",
            "condition_label": "AlloDiverse, original training",
            "dataset": "AlloDiverse",
            "task": ALLODIVERSE_TASKS[emb],
        })
        task_key_rows.append({
            "embedding_key": emb,
            "embedding_label": EMBEDDING_LABELS[emb],
            "condition_key": "allodiverse_balanced",
            "condition_label": "AlloDiverse, balanced training",
            "dataset": "AlloDiverse",
            "task": ALLODIVERSE_TASKS[emb],
        })
        task_key_rows.append({
            "embedding_key": emb,
            "embedding_label": EMBEDDING_LABELS[emb],
            "condition_key": "kinsite",
            "condition_label": "KinSite",
            "dataset": "KinSite",
            "task": KINSITE_TASKS[emb],
        })

    task_key_df = pd.DataFrame(task_key_rows)

    out_xlsx = os.path.join(
        output_dir,
        "supplement_all_models_pr_plotting_data.xlsx",
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="AUPR_summary", index=False)
        mean_curve_df.to_excel(writer, sheet_name="PR_curves_mean", index=False)
        seed_curve_df.to_excel(writer, sheet_name="PR_curves_by_seed", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)
        model_key_df.to_excel(writer, sheet_name="Model_key", index=False)
        embedding_key_df.to_excel(writer, sheet_name="Embedding_key", index=False)
        condition_key_df.to_excel(writer, sheet_name="Condition_key", index=False)
        task_key_df.to_excel(writer, sheet_name="Task_key", index=False)

        if not missing_df.empty:
            missing_df.to_excel(writer, sheet_name="Missing", index=False)

        sheets = {
            "AUPR_summary": summary_df,
            "PR_curves_mean": mean_curve_df,
            "PR_curves_by_seed": seed_curve_df,
            "Config": config_df,
            "Model_key": model_key_df,
            "Embedding_key": embedding_key_df,
            "Condition_key": condition_key_df,
            "Task_key": task_key_df,
        }

        if not missing_df.empty:
            sheets["Missing"] = missing_df

        for sheet_name, df in sheets.items():
            autosize_excel_columns(writer, sheet_name, df)

    print(f"Saved Excel file -> {out_xlsx}")
    print("")
    print("Summary:")
    print(f"  AUPR rows        : {len(summary_df)}")
    print(f"  Mean PR rows     : {len(mean_curve_df)}")
    print(f"  Seed PR rows     : {len(seed_curve_df)}")
    print(f"  Missing rows     : {len(missing_df)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--unbalanced_dir",
        default=DEFAULT_UNBALANCED_DIR,
        help="Directory with flat-format unbalanced prediction .npz files.",
    )

    parser.add_argument(
        "--balanced_dir",
        default=DEFAULT_BALANCED_DIR,
        help="Directory with balanced hierarchical AlloDiverse prediction .npz files.",
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the Excel workbook will be written.",
    )

    args = parser.parse_args()

    export_excel(
        unbalanced_dir=args.unbalanced_dir,
        balanced_dir=args.balanced_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()