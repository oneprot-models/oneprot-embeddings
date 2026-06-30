#!/bin/bash

job_script="train_oneprot_ddp.sbatch"
num_iterations=$1
last_job_id=""
run_name=$(date '+%Y-%m-%d__%H:%M:%S')
run_dir="${run_name}/"

for ((i = 1; i <= num_iterations; i++)); do
    output_dir="${run_name}/train_${i}_out.out"
    error_dir="${run_name}/train_${i}_err.out"
    mkdir -p "${run_name}/"
    echo ${run_name}/

    if [ -z "$last_job_id" ]; then
        # First submission, no dependency
        output=$(sbatch  --output=$output_dir --error=$error_dir "$job_script" "epoch_022_09600.ckpt" $run_dir)
    
    else
        # Subsequent submissions, depend on the completion of the last job

        output=$(sbatch --dependency=afterany:$last_job_id --output=$output_dir --error=$error_dir "$job_script" "${run_dir}/last.ckpt" $run_dir)
    fi
    # Extract job ID
    last_job_id=$(echo $output | grep -oP "\d+")
    echo "Submitted job $last_job_id iteration $i"
done