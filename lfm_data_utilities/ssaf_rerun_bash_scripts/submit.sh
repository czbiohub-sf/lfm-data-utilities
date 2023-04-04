#!/bin/bash

# This bash script is intended for submitting jobs to a GPU slurm node.
# This speeds up running inference, such as when rerunning SSAF on large datasets
# 
# For example, to rerun SSAF on curiosity data I would run:
# 	"sbatch submit.sh python3 ssaf_rerun_all.py ...
#	/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/Uganda_full/curiosity ...
#	'/home/michelle.khoo/autofocus/ulc-malaria-autofocus/trained_models/best.pth' ...
# 	'/home/michelle.khoo/autofocus/curiosity'"



#SBATCH --job-name=ULCMalariaSSAFreruns
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-outputs/slurm-%j.out

env | grep "^SLURM" | sort

if [ $# -lt 1 ];
then
  echo "Missing argument, expected syntax: sbatch submit.sh <command>"
else
  eval "$@"
fi
