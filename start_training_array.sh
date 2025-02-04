#!/bin/bash

job_script="train_oneprot_ddp_hdfml.sbatch"
num_iterations=$1
run_name=$(date '+%Y-%m-%d__%H:%M:%S')
run_dir="/p/scratch/hai_oneprot/checkpoints_221024/manual_ckpts/${run_name}/"

# Create the run directory
mkdir -p "$run_dir"

# Submit the job array
output=$(sbatch --array=1-$num_iterations --output="${run_dir}/train_%a_out.out" --error="${run_dir}/train_%a_err.out" "$job_script" $run_dir)

# Extract job ID
job_id=$(echo $output | grep -oP "\d+")

echo "Submitted job array $job_id with $num_iterations iterations"