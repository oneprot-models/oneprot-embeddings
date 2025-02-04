#!/bin/bash

job_script="train_oneprot_ddp_jedi.sbatch"
num_iterations=$1
last_job_id=""
run_name=$(date '+%Y-%m-%d__%H:%M:%S')
#run_dir="/p/project1/hai_oneprot/$USER/oneprot-main/slurms/manual_ckpts/${run_name}/"
run_dir="/p/scratch/hai_oneprot/checkpoints_221024/manual_ckpts/${run_name}/"

for ((i = 1; i <= num_iterations; i++)); do
    # output_dir="/p/project1/hai_oneprot/$USER/oneprot-main/slurms/${run_name}/train_${i}_out.out"
    # error_dir="/p/project1/hai_oneprot/$USER/oneprot-main/slurms/${run_name}/train_${i}_err.out"
    # mkdir -p "/p/project1/hai_oneprot/$USER/oneprot-main/slurms/${run_name}/"
    output_dir="/p/scratch/hai_oneprot/checkpoints_221024/${run_name}/train_${i}_out.out"
    error_dir="/p/scratch/hai_oneprot/checkpoints_221024/${run_name}/train_${i}_err.out"
    mkdir -p "/p/scratch/hai_oneprot/checkpoints_221024/${run_name}/"

    if [ -z "$last_job_id" ]; then
        # First submission, no dependency
        #output=$(sbatch --output=$output_dir --error=$error_dir "$job_script" "no_ckpt" $run_dir)
        output=$(sbatch --dependency=afterany:22316 --output=$output_dir --error=$error_dir "$job_script" "/p/scratch/hai_oneprot/checkpoints_221024/manual_ckpts/2024-10-24__08:27:58/last-v1.ckpt" $run_dir)
        
        #"/p/project/hai_oneprot/bazarova1/oneprot-main/slurms/manual_ckpts/2024-09-05__21:51:21/last.ckpt" #checkpoint for structure only
    
    else
        # Subsequent submissions, depend on the completion of the last job
        output=$(sbatch --dependency=afterany:$last_job_id --output=$output_dir --error=$error_dir "$job_script" "${run_dir}/last.ckpt" $run_dir)
    fi
    # Extract job ID
    last_job_id=$(echo $output | grep -oP "\d+")
    echo "Submitted job $last_job_id iteration $i"
done