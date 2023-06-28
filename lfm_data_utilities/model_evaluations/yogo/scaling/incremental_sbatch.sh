#! /usr/bin/sh

for i in $(seq 1 15); do
  sbatch scripts/submit_cmd.sh yogo train \
    ../dataset_defs/incremental_set_addition/all-labelled-data-test_$i.yml \
    --epochs 128 --batch-size 64 --iou-weight 5.97586 --label-smoothing 0.14 --learning-rate 0.0005690962636948274 --from-pretrained trained_models/valiant-lion-732/best.pth --weight-decay 0.01930025077473428 \
    --note "all-labelled-data-test $i" --tag "incremental-data-test"
done
