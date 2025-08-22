# lfm-data-utilities

This is a collection of auxilliary data processing tools used for offline data analysis for research and development purposes in Remoscope project: https://www.medrxiv.org/content/10.1101/2024.11.12.24317184v1

The code is organized into self-explanatory directories, but otherwise is not tied together as a package. None of the code in this repo is used on the instrument itself. It was created by the Bioengineering team at Chan Zuckerberg Biohub, San Francisco.

# Installation

In a [virtual environment](https://docs.python.org/3/library/venv.html) run `python3 -m pip install .` from `lfm-data-utilities/lfm_data_utilities/` (i.e where `setup.py`) is located.

To run the SSAF scripts in `lfm-data-utilities/lfm_data_utilities/ssaf_rerun_scripts`, you will also need to install the autofocus package from [this Github repo](https://github.com/czbiohub/ulc-malaria-autofocus).
