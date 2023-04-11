# SSAF rerun scripts

### ssaf_rerun.py
Tool to get all zarr .zips from a collection of datasets and run SSAF on each zarr file.

Inputs:
- Path to folder containing datasets: eg. /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/Uganda_full/curiosity
- Path to SSAF model: eg. /ulc-malaria-autofocus/trained_models/best.pth
- Output folder

The SSAF values for each run will be exported to a separate txt file in the output folder.

For example, the SSAF results for `2023-03-04-041223/2023-03-04-041224_/2023-03-04-041224_.zip`will be exported to `{output folder}/2023-03-04-041224__ssaf.txt`

#### Example usage
`python3 ssaf_rerun_all.py /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/Uganda_full/curiosity /home/michelle.khoo/autofocus/ulc-malaria-autofocus/trained_models/best.pth /home/michelle.khoo/autofocus/curiosity`

### submit.sh
Submits jobs to SLURM with GPU configuration. This speeds up SSAF processing and should be used for large jobs.

Jobs can be submitted using the format `sbatch submit.sh {command to be run}`

#### Example usage
`sbatch submit.sh python3 ssaf_rerun_all.py /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/Uganda_full/curiosity /home/michelle.khoo/autofocus/ulc-malaria-autofocus/trained_models/best.pth /home/michelle.khoo/autofocus/curiosity`