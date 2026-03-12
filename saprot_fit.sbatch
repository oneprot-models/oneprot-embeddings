#!/bin/bash -x

##SBATCH --account profound
##SBATCH --account westai0009
#SBATCH --account loki
##SBATCH --account atmlaml                                                # specify the account name, otherwise the job will not run
#SBATCH --time 1:30:00                                                            # pecify the running time; on develbooster it is limited to 2 hours
#SBATCH --cpus-per-task=4                                                         # do not change
#SBATCH --ntasks-per-node 4                                                        # number of tasks per node; each node has 4 GPUs
#SBATCH --nodes 1                                                                  # the number of nodes; if you request 4 nodes you get 16 GPUs
##SBATCH --partition batch
##SBATCH --partition dc-hwai
#SBATCH --partition dc-gpu
##SBATCH --partition booster
##SBATCH --partition dc-gpu-devel                                                  # the walltime on develbooster is 2 hours, on booster 24 hours; maximum number of nodes on develbooster is 4 nodes
#SBATCH --gres gpu:4
##SBATCH --begin=01:30:00     
#SBATCH --output /p/project1/hai_oneprot/bazarova1/oneprot-refined/slurms/SaProt_fit_%j_out.out  # specify the file for the slurm output; make sure to change it to your folder
#SBATCH --error  /p/project1/hai_oneprot/bazarova1/oneprot-refined/slurms/SaProt_fit_%j_err.out                                                         # do not change
#SBATCH --dependency=afterany:10562913

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
export MASTER_ADDR="${MASTER_ADDR}i"

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
RESULTS_SUFFIX="$1"
TASK_NAME="$2"
MODEL_NAME="$3"


module purge
module load Stages/2024  GCCcore/.12.3.0 Apptainer-Tools/2024
#module load Apptainer-Tools


#echo "-------- Inside of the container ------------"

##################################################################################

# The below script runs a job using the container only
# The hydra setup will use the file train_ddp.yaml, as specified by the comment below; check the hydra documentation to see other option of specifying hydra parameters

#srun --cpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH=""; export HYDRA_FULL_ERROR=1; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif python3 src/train.py experiment=train_ddp'

# The below script activates a virtual environment within the container
#srun --cpu-bind=none bash -c "export CUDA_VISIBLE_DEVICES=\"0,1,2,3\"; export PYTHONPATH=\"\"; export HYDRA_FULL_ERROR=1; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif bash -c \"source /p/project/hai_oneprot/bazarova1/sc_venv_template_erinc/venv/bin/activate && python src/saprot_fit_cls.py \""
#srun --cpu-bind=none bash -c "export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH=""; export HYDRA_FULL_ERROR=1; export LD_LIBRARY_PATH=/p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif:/usr/local/lib/python3.10/dist-packages/xgboost/lib:$LD_LIBRARY_PATH; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif bash -c "source /p/project/hai_oneprot/bazarova1/sc_venv_template_erinc/venv/bin/activate && python src/saprot_fit.py transform_func=$BITVECTOR system=$SYSTEM classifier=$CLASSIFIER task_name=$TASK_NAME\""
#srun --cpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH=""; export HYDRA_FULL_ERROR=1; export LD_LIBRARY_PATH=/p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif:/usr/local/lib/python3.10/dist-packages/xgboost/lib:$LD_LIBRARY_PATH; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif bash -c "source /p/project/hai_oneprot/bazarova1/sc_venv_template_erinc/venv/bin/activate && python src/saprot_fit.py transform_func=$BITVECTOR system=$SYSTEM classifier=$CLASSIFIER task_name=$TASK_NAMEC"'
srun --cpu-bind=none bash -c "export CUDA_VISIBLE_DEVICES=\"0,1,2,3\"; export PYTHONPATH=\"\"; export HYDRA_FULL_ERROR=1; apptainer run --nv /p/project1/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif bash -c \"python3.10 src/saprot_fit_mlp_balanced.py +sweep.task_name=[$TASK_NAME] +sweep.model_type=[$MODEL_NAME] emb_dir=/p/project1/hai_oneprot/bazarova1/oneprot-panda/embeddings/ results_dir=/p/project1/hai_oneprot/bazarova1/oneprot-refined/results_checkpoints_no_sweep_${RESULTS_SUFFIX}/ seed=${RESULTS_SUFFIX}\""


