#! /bin/bash


OUTPUT_DIR_ROOT=/hpc/projects/group.bioengineering/LFM_scope/thumbnail-corrections/high-gametocyte-density

while read line;
do
  run="$(fd "$line" /hpc/projects/group.bioengineering/LFM_scope/scope-parasite-data/run-sets --max-depth 2 -1)"

  sbatch submit_cmd_gpu.sh \
    ./thumbnail_sort_labelling.py create \
    "$OUTPUT_DIR_ROOT/$(basename $line)" \
    --path-to-run "$run" \
    --thumbnail-type yogo-confidence \
    --path-to-pth ~/celldiagnosis/yogo/trained_models/frightful-wendigo-1931/best.pth \
    --overwrite-previous-thumbnails \
    --ignore-class healthy --ignore-class misc --ignore-class wbc;
done < high-gam-list.txt
