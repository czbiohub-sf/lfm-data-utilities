#!/bin/bash

#SBATCH --job-name=ULCMalariaYOGOTraining
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=32
#SBATCH --output=./slurm-outputs/slurm-%j.out

env | grep "^SLURM" | sort

nvcc --version

nvidia-smi

echo "$@"

conda run "$@"
