#!/bin/bash -x

#SBATCH --account XXX
#SBATCH --time 1:00:00                                                            
#SBATCH --cpus-per-task=4                                                         
#SBATCH --ntasks-per-node 4                                                         
#SBATCH --nodes 1
#SBATCH --partition XXX                                                                                                      
#SBATCH --gres gpu:4
#SBATCH --output esm_SaProt_collect%j_out.out 
#SBATCH --error esm_SaProt_collect%j_err.out                                                          


######################## DO NOT CHANGE THE BELOW ################################

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export MASTER_PORT=$((10000 + $RANDOM % 50000))

CACHE_DIR=/huggingface
#export TRANSFORMERS_CACHE="$CACHE_DIR"/models
export TRANSFORMERS_CACHE="$CACHE_DIR"/hub
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE="$CACHE_DIR"/datasets
export HF_MODULES_CACHE="$CACHE_DIR"/modules
export HF_METRICS_CACHE="$CACHE_DIR"/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
export MASTER_ADDR="${MASTER_ADDR}i"

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
export GLOO_SOCKET_IFNAME=eth0  # or the correct interface for your cluster




#echo "-------- Inside of the container ------------"

##################################################################################

# The below script runs a job using the container only
# The hydra setup will use the file train_ddp.yaml, as specified by the comment below; check the hydra documentation to see other option of specifying hydra parameters

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
    python src/collect_embeddings.py single_batch_mode=false" 
    

