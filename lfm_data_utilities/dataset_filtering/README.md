# Dense Run Metadata

(I don't really like the name, please suggest a better one)

This folder contains programs for:

- calculating dense metrics (e.g. autofocus on *every frame* instead of on every 10 frames that occurs on the scope)
- tools to estimate quality of a runset from dense metrics

## Running the tool

By default, if you run `sbatch write_dense_data_csv.sh`, it will

- calculate metrics over `/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/scope-parasite-data/run-sets`
- write results to `/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/scope-parasite-data/dense-data`
- use yogo model `~/celldiagnosis/yogo/trained_models/volcanic-sweep-69/best.pth`
- use autofocus model `~/autofocus/ulc-malaria-autofocus/trained_models/noble-deluge-171/best.pth`

You can modify these as you wish. I should write this more generally.
