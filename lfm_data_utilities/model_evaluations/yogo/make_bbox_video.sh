#! /usr/bin/sh


if [ $# -ne 2 ]; then
  echo "Usage: $0 path-to-pth-file path-to-images"
  exit 1
fi

if [ -d "$2" ]; then
  # if path-to-images is a directory, check that it is only full of pngs
  if [ -z "$(ls -A $2/*.png)" ]; then
    echo "Error: $2 is empty"
    exit 1
  fi
  yogo infer $1 --path-to-images $2 --output-dir /tmp/temporary-yogo-images --draw-boxes
elif [[ $2 == "*.zip" ]]; then
  # elif it is a zip file, assume it is zarr
  yogo infer $1 --path-to-zarr $2 --output-dir /tmp/temporary-yogo-images --draw-boxes
fi

if [ $? -ne 0 ]; then
  exit 1
fi

# check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
  echo "Error: ffmpeg is not installed"
  exit 1
fi

ffmpeg -y -framerate 30 -pattern_type glob -i '/tmp/temporary-yogo-images/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
