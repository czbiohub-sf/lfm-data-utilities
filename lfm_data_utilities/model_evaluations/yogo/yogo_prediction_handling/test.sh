#! /usr/bin/env bash

if [ ! -d "venv" ]; then
  # create virtual environment
  echo "creating virtual environment..."
  python3 -m venv "/hpc/mydata/$USER/lfmdu-yogo-prediction-test-venv"
  ln -s "/hpc/mydata/$USER/lfmdu-yogo-prediction-test-venv" "venv"
fi

echo "activating virtual environment..."
source venv/bin/activate

# if we can't find `yogo` in `python3 -m pip freeze`, install it with deps
# otherwise, just update it with --no-deps
if [ ! -f "venv/bin/yogo" ]; then
  echo "installing yogo..."
  python3 -m pip install --force-reinstall "git+ssh://git@github.com/czbiohub-sf/yogo.git@prediction-handling-v2#egg=yogo" > /dev/null
else
  echo "updating yogo..."
  python3 -m pip install --force-reinstall --no-deps "git+ssh://git@github.com/czbiohub-sf/yogo.git@prediction-handling-v2#egg=yogo" > /dev/null
fi

model_path=~/celldiagnosis/yogo/trained_models/absurd-moon-607/best.pth
data_path=~/celldiagnosis/dataset_defs/human-labels/all-dataset-subsets-no-aug.yml

srun --partition=gpu --gres=gpu:1 -c 16 yogo test $model_path $data_path --wandb --include-mAP --tags handling-test default --prediction-formatter default &
srun --partition=gpu --gres=gpu:1 -c 16 yogo test $model_path $data_path --wandb --include-mAP --tags handling-test opt_A --prediction-formatter opt_A &
srun --partition=gpu --gres=gpu:1 -c 16 yogo test $model_path $data_path --wandb --include-mAP --tags handling-test opt_B --prediction-formatter opt_B &

wait

deactivate
