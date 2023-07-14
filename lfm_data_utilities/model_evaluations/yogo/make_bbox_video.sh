#! /usr/bin/env bash

# if there are 3 args, the last is the output fname
if [ $# -eq 3 ]; then
  out_fname=$3
elif [ $# -eq 2 ]; then
  out_fname="out.mp4"
else
  echo "Usage: $0 path-to-pth-file path-to-images [video-output-name out.mp4] (got $# args)"
  exit 1
fi

# check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
  echo "Error: ffmpeg is not installed"
  exit 1
fi

rm -f /tmp/temporary-yogo-images/*

if [[ -d "$2" ]]; then
  # if path-to-images is a directory, check that it is only full of pngs
  png_count=$(find "$2" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.tiff" \) | wc -l)
  if [ "$png_count" -eq 0 ]; then
    echo "There are no png files in the directory."
    exit 1
  fi
  yogo infer $1 --path-to-images $2 --output-dir /tmp/temporary-yogo-images --draw-boxes --batch-size 128 --output-img-filetype ".tiff"
elif [[ $2 == *.zip ]]; then
  # elif it is a zip file, assume it is zarr
  yogo infer $1 --path-to-zarr $2 --output-dir /tmp/temporary-yogo-images --draw-boxes --batch-size 128 --output-img-filetype ".tiff"
fi

if [ $? -ne 0 ]; then
  exit 1
fi

ffmpeg -y -framerate 30 -pattern_type glob -i '/tmp/temporary-yogo-images/*.tiff' -c:v libx264 -r 30 -crf 18 -pix_fmt yuv420p "$out_fname" < /dev/null || exit
