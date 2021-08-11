#!/usr/bin/env bash
#SBATCH --output=/share/quonlab/workspaces/ricsharma/slurm_output/slurm-out-%j
#SBATCH --error=/share/quonlab/workspaces/ricsharma/slurm_output/slurm-error-%j
#SBATCH -p production
#SBATCH -t 03:00:00
#SBATCH --mem 12000
#SBATCH --gres=gpu:0,gpu_mem:4000
aklog
hostname
nvidia-smi
source ~/.bashrc

## Run model
cd /share/quonlab/workspaces/ricsharma/bpnet_on_gpu/bpnet
conda activate my_bpnet
export NUMEXPR_MAX_THREADS=64
bpnet train dataspec.yml --config=config_oncpu.gin . --run-id 'job_alltracks_1epoch_cpu'
