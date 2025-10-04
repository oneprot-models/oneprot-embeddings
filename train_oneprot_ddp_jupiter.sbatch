#!/bin/bash -x

#SBATCH --account oneprotgpt
##SBATCH --account zam
#SBATCH --time 00:20:00                                                            # pecify the running time; on develbooster it is limited to 2 hours
#SBATCH --cpus-per-task=18                                                         # do not change
#SBATCH --ntasks-per-node 4                                                       # number of tasks per node; each node has 4 GPUs
#SBATCH --nodes 1 # the number of nodes; if you request 4 nodes you get 16 GPUs
#SBATCH --output /e/project1/oneprotgpt/oneprot-panda/%j.out  # specify the file for the slurm output; make sure to change it to your folder
#SBATCH --error /e/project1/oneprotgpt/oneprot-panda/%j_err.out
##SBATCH --partition dc-gh
#SBATCH --partition booster                                                  # the walltime on develbooster is 2 hours, on booster 24 hours; maximum number of nodes on develbooster is 4 nodes
#SBATCH --gres gpu:4
                                                               # do not change

######################## DO NOT CHANGE THE BELOW ################################

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

CACHE_DIR=/p/scratch/hai_oneprot/huggingface
export TRANSFORMERS_CACHE="$CACHE_DIR"/models
export HF_DATASETS_CACHE="$CACHE_DIR"/datasets
export HF_MODULES_CACHE="$CACHE_DIR"/modules
export HF_METRICS_CACHE="$CACHE_DIR"/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
#export MASTER_ADDR="${MASTER_ADDR}i"

export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


#module purge
#module load Stages/2024  GCCcore/.12.3.0 Apptainer-Tools/2024
#module load Apptainer-Tools


#echo "-------- Inside of the container ------------"

##################################################################################

# The below script runs a job using the container only
# The hydra setup will use the file train_ddp.yaml, as specified by the comment below; check the hydra documentation to see other option of specifying hydra parameters

#srun --cpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH=""; export HYDRA_FULL_ERROR=1; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif python3 src/train.py experiment=train_ddp'

# The below script activates a virtual environment within the container

#srun --cpu-bind=none bash -c "export CUDA_VISIBLE_DEVICES=\"0,1,2,3\"; export PYTHONPATH=\"\"; export HYDRA_FULL_ERROR=1; apptainer run --nv -B $(pwd):/p/project1/hai_oneprot/bazarova1/oneprot-refined/ -B /p/scratch/hai_oneprot/huggingface/:/p/scratch/hai_oneprot/huggingface/ -B /p/scratch/hai_oneprot/checkpoints_refined_051024/:/p/scratch/hai_oneprot/checkpoints_refined_051024/ -B /p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/:/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/ /p/scratch/hai_oneprot/container_jedi/oneprot_arm64_final.sif bash -c \"python src/train.py +experiment=train_ddp_1.yaml bash -c \"source /p/scratch/hai_oneprot/container_jedi/alina_singularity/sc_venv_new/venv/bin/activate && python src/train.py +experiment=train_ddp_1.yaml\""
ckpt_path=$1
run_dir=$2

# last_ckpt="$2/last-v1.ckpt"
# if [ -f "$last_ckpt" ]; then
#     mv "$last_ckpt" "$2/last.ckpt"
# fi

export sif_path="/p/data1/profound_data/bazarova1/pyg-25-03-py3.sif" 


#srun --cpu-bind=none bash -c "export CUDA_VISIBLE_DEVICES="0,1,2,3";export PYTHONPATH="";  export HYDRA_FULL_ERROR=1; source /p/project1/hai_oneprot/bazarova1/oneprot-refined/sc_venv_booster/activate.sh && python src/train.py +experiment=train_ddp_10440880.yaml"
#srun --cpu-bind=none bash -c "export CUDA_VISIBLE_DEVICES=\"0,1,2,3\"; export PYTHONPATH=\"/p/project1/hai_oneprot/bazarova1/oneprot-panda\"; export HYDRA_FULL_ERROR=1; apptainer run --nv /e/project1/oneprotgpt/oneprot-panda/container/pyg-25-03-py3.sif bash -c \"source /e/project1/oneprotgpt/sc_venv_2503/venv/bin/activate && python /p/project1/hai_oneprot/bazarova1/oneprot-panda/src/train_oneprot.py +experiment=train_ddp_allatom.yaml ckpt_path=${ckpt_path}  callbacks.model_checkpoint.dirpath=${run_dir}\"" # --cfg job --resolve\"" #  ckpt_path=${ckpt_path} callbacks.model_checkpoint.dirpath=${run_dir}\"" 


srun --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES=\"0,1,2,3\"; \
    export PYTHONPATH=\"\"; \
    export HYDRA_FULL_ERROR=1; \
    apptainer run --nv \
        -B /e/project1/oneprotgpt/oneprot-panda/:/e/project1/oneprogpt/oneprot-panda/ \
        -B /p/scratch/oneprotgpt:/p/scratch/oneprotgpt \
        -B /p/scratch/hai_oneprot/checkpoints_refined_111024/:/p/scratch/hai_oneprot/checkpoints_refined_111024/ \
        -B /p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/:/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/ \
        -B /e/project1/oneprotgpt/sc_venv_2503/:/e/project1/oneprotgpt/sc_venv_2503/ \
        /e/project1/oneprotgpt/oneprot-panda/container/pyg-25-03-py3.sif \
        bash -c \"
            source /e/project1/oneprotgpt/sc_venv_2503/venv/bin/activate && \
            python src/train_oneprot.py +experiment=train_ddp_allatom.yaml\"" #ckpt_path=${ckpt_path} callbacks.model_checkpoint.dirpath=${run_dir}
 #       \"
#"

# srun --cpu-bind=none bash -c "
#     export CUDA_VISIBLE_DEVICES=\"0,1,2,3\"; \
#     export PYTHONPATH=\"\"; \
#     export HYDRA_FULL_ERROR=1; \
#     apptainer run --nv \
#         -B $(pwd):/p/project1/hai_oneprot/bazarova1/oneprot-refined/ \
#         -B /p/scratch/hai_oneprot/huggingface/:/p/scratch/hai_oneprot/huggingface/ \
#         -B /p/scratch/hai_oneprot/checkpoints_refined_111024/:/p/scratch/hai_oneprot/checkpoints_refined_111024/ \
#         -B /p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/:/p/scratch/hai_oneprot/merdivan1/pretrain_dataset/50ss/ \
#         -B /p/scratch/hai_oneprot/container_jedi/alina_singularity/sc_venv_new/:/p/scratch/hai_oneprot/container_jedi/sc_venv_nv/ \
#         -B /p/scratch/hai_oneprot/checkpoints_refined_101024/2024-10-11_09-53-57/:/p/scratch/hai_oneprot/checkpoints_refined_101024/2024-10-11_09-53-57/ \
#         /p/scratch/hai_oneprot/container_jedi/oneprot_arm64_230_final.sif \
#         bash -c \"
#             source /p/scratch/hai_oneprot/container_jedi/sc_venv_nv/venv/bin/activate && \
#             python src/train.py +experiment=train_ddp_2.yaml
#         \"
# "


#srun --cpu-bind=none bash -c "export CUDA_VISIBLE_DEVICES="0,1,2,3";export PYTHONPATH="";  export HYDRA_FULL_ERROR=1; source /p/project/hai_oneprot/bazarova1/sc_venv_template_for_unimol/activate.sh && python src/train.py experiment=train_ddp"

#srun --cpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH="";export HYDRA_FULL_ERROR=1; \
# apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif bash -c \
#  "source /p/project/hai_oneprot/bazarova1/sc_venv_template_for_unimol/activate.sh && python src/train.py experiment=train_ddp"'
