#! /usr/bin/env bash


if [ ! -d "venv" ]; then
  # create virtual environment
  echo "creating virtual environment..."
  python3 -m venv "/hpc/mydata/$USER/lfmdu-yogo-prediction-test-venv"
  ln -s "/hpc/mydata/$USER/lfmdu-yogo-prediction-test-venv" "venv"
fi

echo "activating virtual environment..."
source venv/bin/activate

echo "installing yogo..."
python3 -m pip install git+ssh://git@github.com/czbiohub-sf/yogo.git@prediction-handling-v2#egg=yogo

deactivate
