#! /bin/bash

exec 'source ~/.bashrc'
exec 'conda activate ssaf'


if [ $# -lt 2 ];
then
  echo "Missing argument, expected syntax: source rerun_ssaf_all_datasets.sh <scope folder> <path to ulc-malaria-autofocus repo>"
else
  file_format='*-*-*-*_.zip'
  #find $data_dir -path '$path_format' -name '$file_format' -exec bash -c 'sbatch submit.sh python3 $autofocus_dir/infer.py $autofocus_dir/trained_models/best.pth --zarr {} --output $(dirname {})/$(basename {} .zip)ssaf.txt' \;
  find $data_dir -path '$path_format' -name '$file_format' -exec bash -c 'sbatch submit.sh python3 $autofocus_dir/infer.py $autofocus_dir/trained_models/best.pth --zarr {} --output ~/autofocus/ssaf-outputs/$(basename {} .zip)ssaf.txt' \;

fi

echo "Finished processing"
