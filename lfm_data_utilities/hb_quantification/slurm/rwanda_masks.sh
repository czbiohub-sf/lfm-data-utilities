#!/bin/bash

#SBATCH --job-name=rwanda_masks
#SBATCH --output=rwanda_masks.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=64

python3 ../get_masks.py <<< "../inputs/rwanda_mch_data.csv"
