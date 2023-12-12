#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1         # This needs to match Trainer(num_nodes=...)
#SBATCH --account=euge-k
#SBATCH --partition=gilbreth-k
#SBATCH --gres=gpu:1        # Number of GPUs per node | This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1  
#SBATCH --time=2-00:00:00
#SBATCH --job-name=nougart_embbedings  # Change the name to match the job
#SBATCH --output=slurmout/%x-%j.out
#SBATCH --error=slurmout/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=viktor.ciroski@purdue.edu

# Activate conda environment
module load anaconda/2020.11-py38
module load use.own
conda activate elab_2

# Set environment variables
export RANK=2              # Set the rank for each task
export WORLD_SIZE=4        # Set the total number of tasks

echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE" 

export CUDA_VISIBLE_DEVICES=0 
export TRANSFORMERS_CACHE=/depot/euge/etc/models
export HF_HOME=/depot/euge/etc/models
export HUGGING_FACE_HUB_TOKEN="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"

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster, you might need these:
# Set the network interface
export NCCL_SOCKET_IFNAME="ib"

# Run the script using torchrun
srun python main.py
