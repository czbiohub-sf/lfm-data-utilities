#! /usr/bin/env bash


# for each of the runs here:
#
# /hpc/projects/group.bioengineering/LFM_scope/misc/uganda-fp-thumbnails/frightful-wendigo-1931_false_positives.csv
# converted and cleaned to get
# /hpc/projects/group.bioengineering/LFM_scope/misc/uganda-fp-thumbnails/frightful-wendigo-1931_false_positive_runs.txt

while read run_folder; do
  run_name=$(basename $run_folder)

  ./thumbnail_sort_labelling.py create \
    "/hpc/projects/group.bioengineering/LFM_scope/misc/uganda-fp-thumbnails/$run_name" \
    --path-to-run "$run_folder" \
    --ignore-class healthy --ignore-class misc --ignore-class wbc \
    --thumbnail-type yogo-confidence \
    --min-confidence 0.9 \
    --obj-thresh 0.5 \
    --iou-thresh 0.5 \
    --path-to-pth /hpc/projects/group.bioengineering/LFM_scope/yogo_models/all-models/frightful-wendigo-1931/best.pth
done < /hpc/projects/group.bioengineering/LFM_scope/misc/uganda-fp-thumbnails/frightful-wendigo-1931_false_positive_runs.txt
