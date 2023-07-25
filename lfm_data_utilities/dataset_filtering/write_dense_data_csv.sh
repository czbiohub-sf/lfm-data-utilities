#! /bin/bash

#SBATCH --job-name=DenseData
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=64
#SBATCH --output=./slurm-outputs/slurm-%j.out


ulimit -H -c unlimited

conda run ./create_dense_data.py \
  /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/Uganda_full_2 \
  /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/Uganda_full_2/dense-data \
  /home/axel.jacobsen/celldiagnosis/yogo/trained_models/devoted-snowflake-1666/best.pth \
  /home/axel.jacobsen/autofocus/ulc-malaria-autofocus/trained_models/solar-microwave-438/best.pth
