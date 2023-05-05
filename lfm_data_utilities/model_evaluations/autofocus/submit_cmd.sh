#!/bin/bash

#SBATCH --job-name=ConfusionMatrixConstruction
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --output=./%j.out

env | grep "^SLURM" | sort

nvcc --version

nvidia-smi

echo "$@"

conda run "$@"
