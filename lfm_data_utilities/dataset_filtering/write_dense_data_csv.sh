#! /bin/bash

#SBATCH --job-name=DenseData
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --output=./slurm-outputs/%j.out

env | grep "^SLURM" | sort

echo "writing dense data w/ args"

SOURCE_DIR="/hpc/projects/group.bioengineering/LFM_scope/Uganda_full_2"
TARGET_DIR="/hpc/projects/group.bioengineering/LFM_scope/Uganda_full_2_images/dense-data"
YOGO_MODEL="/hpc/projects/group.bioengineering/LFM_scope/celldiagnosis/yogo/trained_models/frightful-wendigo-1931/best.pth"
AUTOFOCUS_MODEL="/hpc/projects/group.bioengineering/LFM_scope/autofocus/ulc-malaria-autofocus/trained_models/polished-dragon-468/best.pth"

echo "SOURCE_DIR is $SOURCE_DIR"
echo "TARGET_DIR is $TARGET_DIR"
echo "YOGO_MODEL is $YOGO_MODEL"
echo "AUTOFOCUS_MODEL is $AUTOFOCUS_MODEL"

ulimit -H -c unlimited

conda run ./create_dense_data.py \
  $SOURCE_DIR \
  $TARGET_DIR \
  $YOGO_MODEL \
  $AUTOFOCUS_MODEL
