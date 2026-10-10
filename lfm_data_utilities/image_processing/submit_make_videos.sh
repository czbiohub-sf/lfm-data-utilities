#!/bin/bash
#SBATCH --job-name=make_video
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=logs/make_video_%A_%a.out
#SBATCH --error=logs/make_video_%A_%a.err

# Usage (run directly, NOT via sbatch - it submits itself as an array job):
#   fd -ae zip | ./submit_make_videos.sh [--contrast-boost] [--no-img-counter]
#   ./submit_make_videos.sh [make_video.py flags] <zarr.zip or folder> ...
# One array task per unique dataset folder; videos are saved next to each zarr.
# MAX_PARALLEL (default 50) caps simultaneously running tasks.

REPO=/hpc/user_apps/bioengineering/lfm-data-utilities

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    FLAGS=()
    PATHS=()
    for a in "$@"; do
        if [[ "$a" == --* ]]; then FLAGS+=("$a"); else PATHS+=("$a"); fi
    done
    [ ${#PATHS[@]} -eq 0 ] && [ ! -t 0 ] && mapfile -t PATHS

    mkdir -p logs
    LIST="$PWD/logs/make_video_paths_$(date +%Y%m%d-%H%M%S).txt"
    for p in "${PATHS[@]}"; do
        p=$(realpath "$p")
        [ -f "$p" ] && p=$(dirname "$p")
        echo "$p"
    done | sort -u > "$LIST"

    N=$(wc -l < "$LIST")
    if [ "$N" -eq 0 ]; then
        echo "No paths given. Usage: fd -ae zip | $0 [--contrast-boost]"
        exit 1
    fi
    echo "Submitting $N tasks (list: $LIST)"
    exec sbatch --array=1-"$N"%"${MAX_PARALLEL:-50}" "$0" "$LIST" "${FLAGS[@]}"
fi

LIST="$1"; shift
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "Task $SLURM_ARRAY_TASK_ID on $(hostname): $DIR"

source "$REPO/.venv/bin/activate"
cd "$REPO/lfm_data_utilities/image_processing"

START=$SECONDS
python make_video.py "$@" "$DIR"
STATUS=$?
ELAPSED=$((SECONDS - START))
printf "Task %s finished (exit %d) in %dm%02ds: %s\n" \
    "$SLURM_ARRAY_TASK_ID" "$STATUS" $((ELAPSED / 60)) $((ELAPSED % 60)) "$DIR"
exit $STATUS
