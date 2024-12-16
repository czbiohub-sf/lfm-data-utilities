#! /usr/bin/sh

for i in $(seq 1 17); do
  sbatch submit_cmd.sh yogo train \
    /hpc/projects/group.bioengineering/LFM_scope/celldiagnosis/dataset_defs/incremental_set_addition/all-labelled-$i-healthy.yml \
    --epochs 32 \
    --batch-size 64 \
    --iou-weight 5.97586 \
    --label-smoothing 0.14 \
    --learning-rate 0.0005690962636948274 \
    --from-pretrained /hpc/projects/group.bioengineering/LFM_scope/celldiagnosis/yogo/trained_models/valiant-lion-732/best.pth \
    --weight-decay 0.01930025077473428 \
    --note "all-labelled-data-test-healthy $i" --tag "incremental-data-test-healthy"
done
