#!/bin/bash

#SBATCH --job-name=zstack_mch
#SBATCH --output=zstack_mch.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=64

python3 ../get_zstack_mch.py <<< $'/hpc/projects/group.bioengineering/LFM_scope/SingleShotAutofocus/unsorted/perseverance_and_zenith_20240910/train/2024-09-10-152146-local_zstack/\n408\n20'
