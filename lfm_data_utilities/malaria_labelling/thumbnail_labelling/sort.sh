#! /bin/bash

# first argument is ddfs dir
ddfs_dir=$1

for f in $(ls -1 "$ddfs_dir"); do
  # replace .yml in $f with ""
  dirname=${f%.yml}
  outdir="/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/thumbnail-corrections/Uganda-subsets/$dirname"
  mkdir -p outdir

  ./thumbnail_sort_labelling.py create-thumbnails \
      "$outdir" \
      --path-to-labelled-data-ddf "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/Uganda_full_images/subsets-for-labelling/ddfs/$f"
done
