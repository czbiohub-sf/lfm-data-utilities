#! /bin/bash

# executed in lfm-data-utilities/lfm_data_utilities/model_evaluations/yogo
# layer-mosaic.py is in that dir
# The data dir just has some runs:
#   $ ls
#   07529/          README.md       high-titration/ low-titration/


BASE_DIR="/hpc/projects/group.bioengineering/LFM_scope/yogo-layer-analysis-dataset"
HIGH_TITRATION_DIR="$BASE_DIR/high-titration"
LOW_TITRATION_DIR="$BASE_DIR/low-titration"

process_png() {
  MODEL_PATH="/hpc/projects/group.bioengineering/LFM_scope/yogo/trained_models/still-voice-4405/best.pth"

  local png_file=$1
  local tiff_dir=$2

  local filename="$(basename $png_file .png)"

  local output_dir="$tiff_dir/$filename"
  mkdir -p "$output_dir"

  ./layer-mosaic.py "$png_file" "$MODEL_PATH" --output-dir "$output_dir" --tiff

  echo "processed $filename"
}

export -f process_png

find "$HIGH_TITRATION_DIR" -name '*.png' | \
    parallel process_png {} "$HIGH_TITRATION_DIR/2023-08-21-121718_-tiffs"

find "$LOW_TITRATION_DIR" -name '*.png' | \
    parallel process_png {} "$LOW_TITRATION_DIR/2023-08-21-172302_-tiffs"
