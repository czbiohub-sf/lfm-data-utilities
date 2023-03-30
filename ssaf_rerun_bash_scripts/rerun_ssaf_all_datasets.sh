#! /bin/bash

if [ $# -eq 0 ];
then
  echo "Missing argument, expected syntax: source rerun_ssaf_all_datasets.sh <scope folder>"
else
# Make sure globstar is enabled
shopt -s globstar
for i in **/*.txt; do # Whitespace-safe and recursive
    process "$i"
done
  # for file in "$1/"*
  #   do
  #     [[ -d "$file" ]] && echo "$file is a directory"
  #     [[ -f "$file" ]] && echo "$file is a regular file"
  # done
fi

echo "And then we do something with $1"