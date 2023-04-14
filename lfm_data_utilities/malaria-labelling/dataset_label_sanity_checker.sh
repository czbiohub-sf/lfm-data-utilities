#!/bin/bash

count=0

find . -type d -name "images" > /tmp/img_dirs.txt

while read images_dir; do
  # Check if the corresponding labels directory exists
  labels_dir="$images_dir/../labels"
  if [ ! -d "$labels_dir" ]; then
    echo "dANG"
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
