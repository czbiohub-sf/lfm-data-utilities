#! /bin/bash

#SBATCH --job-name=DenseData
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=64
#SBATCH --output=./slurm-outputs/slurm-%j.out


ulimit -H -c unlimited

conda run ./create_dense_data.py \
  /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/scope-parasite-data/run-sets \
  /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/scope-parasite-data/dense-data \
  ~/celldiagnosis/yogo/trained_models/old-transport-1520/best.pth \
  ~/autofocus/ulc-malaria-autofocus/trained_models/valiant-disco-119/best.pth
