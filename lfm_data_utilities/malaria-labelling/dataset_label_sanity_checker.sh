#!/bin/bash

count=0

# argument 1 should be the parent directory to search from
# argument 2 should be the name of the labels dir - default to "labels"

# if there are not 1 or two arguments, print usage and exit
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <parent_dir> [labels_dir]"
  exit 1
fi

parent=$1

if [ $# -eq 2 ]; then
  labels_dir_name=$2
else
  labels_dir_name="labels"
fi

find "$parent" -type d -name "images" > /tmp/img_dirs.txt

while read images_dir; do
  # Check if the corresponding labels directory exists
  labels_dir="$images_dir/../$labels_dir_name"
  if [ ! -d "$labels_dir" ]; then
    continue
  fi

  # Count number of files in images and labels
  num_images=$(ls -1 "$images_dir" | wc -l)
  num_labels=$(ls -1 "$labels_dir" | wc -l)

  # Compare the counts and print the run folder name if they are not equal
  if [ "$num_images" -ne "$num_labels" ]; then
    count=$((count + 1))
    echo $(realpath $labels_dir) $num_images $num_labels
  fi
done < /tmp/img_dirs.txt

echo $count
