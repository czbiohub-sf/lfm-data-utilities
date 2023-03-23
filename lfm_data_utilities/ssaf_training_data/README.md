# Tool for generating SSAF training data
This tool (`main.py`) is used to generate training data for the single shot autofocus model.

### Usage
1. Collect a bunch of local zstacks using `dev_run.py`'s local-zstack button. This will scan through +/- 30 steps of the starting point and take 30 images at each step. Make sure to only press this button once you're at the peak focus position (do this manually or using the existing autofocus model).
2. Run `main.py`. This will ask you for three paths:
    - Path where all the local zstacks are stored
    - Path where the training data should be stored (you can point this to an existing training data folder and it the script will append the new images to the existing folders). 
    - - _The script will create the folder if it doesn't exist already._
    - Path where the focus graphs will be saved (optional, mainly for you to sanity check that the plots look ok).
    - - _The script will create the folder if it doesn't exist already._

### TODO
- Do some fancy CUDA/numba whatever to make the focus metrics calculation faster (look under `utils.py`, `log_power_spectrum_radial_average_sum()`)
- `dev_run.py`: add a checkbox to allow this to run continuously and adjust focus in between acquisitions (using SSAF). The user will still need to intervene to swap out flow cells / adjust hematocrit / etc. to get a sufficient diversity of data.