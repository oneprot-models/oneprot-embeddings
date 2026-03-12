#!/usr/bin/env python3
"""
Calculate average and standard deviation of test_f1_max for each model_type
across multiple result files.
"""

import pandas as pd
import numpy as np
import sys
import os
import torch

# Collect all files
name='merged_pocket_binary_text_comp'
files = [f'/p/project1/hai_oneprot/bazarova1/oneprot-refined/results_checkpoints_no_sweep_/downstream_results/{name}_MLP_results_final_valid.csv']
for i in range(1, 10):
    files.append(f'/p/project1/hai_oneprot/bazarova1/oneprot-refined/results_checkpoints_no_sweep_{i}/downstream_results/{name}_MLP_results_final_valid.csv')

# Read all files and collect data
all_data = []
found_files = []

for f in files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f, skipinitialspace=True)
            df['file'] = f
            all_data.append(df)
            found_files.append(f)
            print(f"✓ Loaded {f}")
        except Exception as e:
            print(f"✗ Error reading {f}: {e}")
    else:
        print(f"✗ Not found: {f}")

if not all_data:
    print("\n❌ No files could be read!")
    print("\nMake sure you run this script from the directory containing:")
    print("  - results_checkpoints_no_sweep_/")
    print("  - results_checkpoints_no_sweep_1/")
    print("  - results_checkpoints_no_sweep_2/")
    print("  - etc.")
    sys.exit(1)

print(f"\n✓ Successfully loaded {len(found_files)} files\n")

# Combine all data
combined = pd.concat(all_data, ignore_index=True)

# normalize column names (remove trailing/leading spaces)
combined.columns = combined.columns.str.strip()

# compute normalized TP/TN columns
a=torch.load(f'embeddings/oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity/{name}/test/{name}_test_embeddings_labels.pt')
norm_tp=float(a['labels_fitness'].sum())
norm_tn=float(len(a['labels_fitness']))
#norm_tp=norm_tn

if 'test_tp' in combined.columns:
    combined['test_tp_norm'] = combined['test_tp'].astype(float) / norm_tp
else:
    combined['test_tp_norm'] = np.nan

if 'test_tn' in combined.columns:
    combined['test_tn_norm'] = combined['test_tn'].astype(float) / norm_tn
else:
    combined['test_tn_norm'] = np.nan

# Group by model_type and calculate stats for test_auc and normalized tp/tn
auc_stats = combined.groupby('model_type')['test_auc'].agg(['mean', 'std', 'count']).rename(columns={'mean':'Average_auc','std':'Std Dev_auc','count':'N'})
tp_stats = combined.groupby('model_type')['test_tp_norm'].agg(['mean', 'std']).rename(columns={'mean':'Average_test_tp_norm','std':'Std Dev_test_tp_norm'})
tn_stats = combined.groupby('model_type')['test_tn_norm'].agg(['mean', 'std']).rename(columns={'mean':'Average_test_tn_norm','std':'Std Dev_test_tn_norm'})

# Merge stats into one table
stats = auc_stats.join(tp_stats).join(tn_stats)

print("=" * 80)
print("Statistics by model_type:")
print("=" * 80)
print(stats.to_string(float_format="{:0.4f}".format))
print("=" * 80)

# Also show individual values for verification
# print("\nIndividual values per file:")
# print("=" * 80)
# for model in combined['model_type          '].unique():
#     print(f"\n{model}:")
#     model_data = combined[combined['model_type          '] == model][['test_f1_max         ', 'file']]
#     for idx, row in model_data.iterrows():
#         filename = os.path.basename(os.path.dirname(os.path.dirname(row['file'])))
#         print(f"  {filename}: {row['test_f1_max         ']:.3f}")