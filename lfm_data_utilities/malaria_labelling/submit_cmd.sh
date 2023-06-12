#!/bin/bash

#SBATCH --job-name=MalariaLabelling
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --output=./slurm-outputs/slurm-%j.out

env | grep "^SLURM" | sort

nvcc --version

nvidia-smi

echo "$@"

conda run "$@"
