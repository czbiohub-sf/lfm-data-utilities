#!/bin/bash

#SBATCH --job-name=heatmapGeneration
#SBATCH --output=slurm-outputs/array/%A_%a.out
#SBATCH --error=slurm-outputs/array/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-1022%16
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1


if [ -z "$1" ]; then
  echo "No path to list of zarr files provided"
  echo "usage: $0 path-to-zarr-list"
  exit 1
fi

ZARR_PATH=$(fd zip $(sed -n "$SLURM_ARRAY_TASK_ID"p "$1") -1)

echo "ZARR_PATH: $ZARR_PATH"

out=$(
  ./create_heatmaps_and_masks.py \
    ~/celldiagnosis/yogo/trained_models/confused-river-1979/best.pth \
    /hpc/projects/group.bioengineering/LFM_scope/Uganda_heatmaps/confused-river-1979 \
    --target-zip "$ZARR_PATH" \
)

if [ $? -eq 0 ]; then
  echo "Successfully created heatmaps and masks on $IMAGES_PARENT_DIR_PATH"
else
  echo "Error occurred during inference on $IMAGES_PARENT_DIR_PATH" >&2
  echo "$out" >&2
fi
