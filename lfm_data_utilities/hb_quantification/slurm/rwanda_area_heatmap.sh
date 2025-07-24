#!/bin/bash

#SBATCH --job-name=rwanda_area_heatmap
#SBATCH --output=rwanda_area_heatmap.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=64

python3 ../load_masks.py
