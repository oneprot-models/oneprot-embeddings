TASKS=(
ASD_merged_pocket_binary_comp
ASD_merged_pocket_binary_text_comp
ASD_merged_pocket_sequence_binary_comp
ASD_merged_pocket_sequence_binary_text_comp
# ASD_pockets_binary_comp
# ASD_pockets_binary_text_comp
# ASD_pockets_sequence_binary_comp
# ASD_pockets_sequence_binary_text_comp
# merged_pocket_binary_comp
# merged_pocket_binary_text_comp
# merged_pocket_sequence_binary_comp
# merged_pocket_sequence_binary_text_comp
)

MODELS=(
        oneprot_pocket_text_32900
        oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity
        oneprot_md_combined_gpcr_no_struct_graph_32900  
        oneprot_md_combined_gpcr_no_struct_token_32900
        oneprot_full_allatom_no_seqsim_no_l1_A100_32900_sanity
        oneprot_pocket_text_32900
        oneprot_struct_graph_pocket_text_32900
        oneprot_md_combined_gpcr_32900
        oneprot_struct_token_pocket_text_32900
)

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Submitting $task | $model"
    sbatch saprot_fit.sbatch "" "$task" "$model"; for i in {1..9}; do sbatch saprot_fit.sbatch $i "$task" "$model"; done
  done
done

# for task in "${TASKS[@]}"; do
# bash process_results_valid.sh "$task"_MLP; for i in {1..9};do bash process_results_valid.sh "$task"_MLP; done
# done