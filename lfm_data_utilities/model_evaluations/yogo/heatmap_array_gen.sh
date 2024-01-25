#!/bin/bash

#SBATCH --job-name=heatmapGeneration
#SBATCH --output=slurm-outputs/array/%A_%a.out
#SBATCH --error=slurm-outputs/array/%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-1022
#SBATCH --cpus-per-task=16
#SBATCH --partition=preempted
#SBATCH --gpus-per-node=1


if [ "$#" -lt 2 ]; then
  echo "No path to list of zarr files provided"
  echo "usage: $0 path-to-zarr-list yogo-model-name"
  exit 1
fi

# ZARR_PATH=$(fd zip $(sed -n "$SLURM_ARRAY_TASK_ID"p "$1") -1)
ZARR_PATH=$(sed -n "$SLURM_ARRAY_TASK_ID"p "$1")
YOGO_MODEL_NAME="$2"

echo "ZARR_PATH: $ZARR_PATH"

out=$(
  ./create_heatmaps_and_masks.py \
    ~/celldiagnosis/yogo/trained_models/$YOGO_MODEL_NAME/best.pth \
    /hpc/projects/group.bioengineering/LFM_scope/Uganda_heatmaps/$YOGO_MODEL_NAME \
    --target-zip "$ZARR_PATH" \
)

if [ $? -eq 0 ]; then
  echo "Successfully created heatmaps and masks on $IMAGES_PARENT_DIR_PATH"
else
  echo "Error occurred during inference on $IMAGES_PARENT_DIR_PATH" >&2
  echo "$out" >&2
fi
