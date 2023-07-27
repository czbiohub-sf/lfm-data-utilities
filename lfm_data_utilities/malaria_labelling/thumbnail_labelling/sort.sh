#! /bin/bash

# first argument is text file of run paths
runs_file=$1
# second argument is output dir
outdir=$2

# if there are not exactly 2 arguments, exit
if [ "$#" -ne 2 ]; then
  echo "usage: sort.sh <runs_text_file> <outdir>"
  exit 1
fi

while IFS= read -r run; do
  echo "processing $run"

  # if run doesn't exist, exit
  if [ ! -d "$run" ]; then
    echo "run $run doesn't exist"
    exit 1
  fi

  folder_outdir="$outdir/$(basename $run)"

  mkdir -p folder_outdir

  ./thumbnail_sort_labelling.py create-thumbnails \
      "$folder_outdir" \
      --path-to-run "$run" \
      --ignore-class "healthy" \
      --ignore-class "misc" \
      --ignore-class "wbc"

done < "$runs_file"
