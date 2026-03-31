#!/bin/bash
#SBATCH -J pretrain_1024_masking_ratio_50
#SBATCH -p <YOUR_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --chdir=<POYO_CAP_ROOT>/scripts
#SBATCH --output=../logs/slurm/%A_%x.txt
#SBATCH --comment=python

# ── Environment Setup ──────────────────────────────────────────────────────
# Adjust the following lines to match your cluster environment.
module purge
# module load cuda/12.1        # uncomment and adjust CUDA version
# source <YOUR_CONDA_INIT>     # e.g. source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <YOUR_ENV>    # e.g. conda activate poyo_cap
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# NCCL settings — adjust for your cluster network topology.
export OMP_NUM_THREADS=16
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# W&B Configuration — set your own credentials via environment variables.
export WANDB_PROJECT=poyo_cap
# export WANDB_ENTITY=<YOUR_WANDB_ENTITY>
# export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

srun python -u ../train.py \
  --config-name pretrain_with_NTSR1_dim_1024.yaml \
  data_root=<YOUR_DATA_ROOT>/allen_brain_observatory_calcium
