#! /bin/bash

#SBATCH --job-name=thumbnail-creation
#SBATCH --output=temp_output/logs/%A_%a.out
#SBATCH --error=temp_output/logs/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-15
#SBATCH --cpus-per-task=8

# check if there is an input
if [ -z "$1" ]; then
  echo "usage: $0 <path-to-file> <output-dir>"
  exit 1
fi

mkdir -p temp_output/logs

# first argument is text file of run paths
run=$(sed -n "$SLURM_ARRAY_TASK_ID"p "$1")
# second argument is output dir
outdir=$2

echo "processing $run"

# if run doesn't exist, exit
if [ ! -d "$run" ]; then
  echo "run $run doesn't exist"
  exit 1
fi

folder_outdir="$outdir/$(basename $run)"

mkdir -p folder_outdir

conda run ./thumbnail_sort_labelling.py create-thumbnails \
    "$folder_outdir" \
    --path-to-run "$run" \
    --ignore-class "healthy" \
    --ignore-class "misc" \
    --ignore-class "wbc"
