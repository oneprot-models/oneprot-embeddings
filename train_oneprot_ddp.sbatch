#!/bin/bash -x

#SBATCH --account XXX
#SBATCH --time 0:30:00                                                            
#SBATCH --cpus-per-task=12                                                         
#SBATCH --ntasks-per-node 4                                                        
#SBATCH --nodes 64 
#SBATCH --output slurms/%j.out  
#SBATCH --error slurms/%j_err.out
#SBATCH --partition XXX
#SBATCH --gres gpu:4
                                                    # do not change

######################## DO NOT CHANGE THE BELOW ################################

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

CACHE_DIR=huggingface
export TRANSFORMERS_CACHE="$CACHE_DIR"/models
export HF_DATASETS_CACHE="$CACHE_DIR"/datasets
export HF_MODULES_CACHE="$CACHE_DIR"/modules
export HF_METRICS_CACHE="$CACHE_DIR"/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3


MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
export MASTER_ADDR="${MASTER_ADDR}i"

export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10

export PROJECT_ROOT='oneprot-embeddings'





#echo "-------- Inside of the container ------------"

##################################################################################

ckpt_path=$1
run_dir=$2

last_ckpt="$2/last-v1.ckpt"
if [ -f "$last_ckpt" ]; then
    mv "$last_ckpt" "$2/last.ckpt"
fi

export sif_path="oneprot_pyg-25-08-py3-amd64.sif"



srun --cpu-bind=none --export=ALL env -u CUDA_VISIBLE_DEVICES \
  apptainer exec --nv \
    --bind logs_oneprot_gpt:/root_dir:rw \
    --bind pretrain_dataset/50ss:/data_dir:ro \
    --bind .:/work_dir:ro \
    --writable-tmpfs \
    "$sif_path" \
  bash -lc "export PYTHONNOUSERSITE=1; unset PYTHONPATH PYENV_VERSION PYENV_ROOT PYENV; \
    source sc_venv_panda/venv/bin/activate && \
    python -c 'import sys, os; print(\"sys.executable:\", sys.executable); print(\"sys.path:\", sys.path[:10])' && \
    python src/train_oneprot.py +experiment=train_ddp_md_combined_3.yaml"



