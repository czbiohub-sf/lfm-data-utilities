#!/bin/bash

#SBATCH --job-name=positional_rwanda_mch
#SBATCH --output=positional_rwanda_mch.out
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=64

python3 ../get_positional_mch.py <<< "../inputs/rwanda_mch_data.csv"
