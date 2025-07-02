#!/bin/bash

#SBATCH --job-name=rwanda_mch
#SBATCH --output=rwanda_mch.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=64

python3 ../cellpose_hb.py <<< "../inputs/rwanda_mch_data.csv"
