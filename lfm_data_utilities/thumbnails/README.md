# Description

## cut_out_cells_using_existing_bb_annotations.py
This script produces 
Two required positional arguments
- EXPERIMENTS_PATH
- SAVE_LOC
    
One optional, explicit argument
- `--label_path` 
    - If explicitly passed the script defaults to searching through all the folders in `LFM_scope/biohub-labels`

### Examples (running from inside the HPC)
(the paths are truncated here for readability)

**Run for all the experiments in a top-level folder**

`python3 cut_out_cells_using_existing_bb_annotations.py scope-parasite-data/run-sets /hpc/mydata/ilakkiyan.jeyakumar/new_folder`

- Go through `scope-parasite-data` and find all the folders which contain an `images/` folder. 
- Go through `LFM_scope/biohub-labels` find the corresponding folders in `biohub-labels`
    - Corresponding folders are found by checking for string matches of `{dataset_name}*/`
- If a folder in `scope-parasite-data` doesn't have a corresponding folder in `biohub-labels`, it is skipped and no thumbnails are made.
- In the command above, the thumbnails are saved to `/hpc/mydata/ilakkiyan.jeyakumar/new_folder`
    - If `new_folder` doesn't already exist, the script will create that folder for you


**Run for a specific folder**
`python3 cut_out_cells_using_existing_bb_annotations.py scope-parasite-data/run-sets/specific_experiments/images biohub-labels/vetted/specific_folder/labels`

- In this case, a single experiment images folder was passed in along with its corresponding labels folder.
- **NOTE**: The script will check that the passed in `images/` and `labels/` folder correspond based on the their parent folder names. **If they don't match, it will warn you that something is amiss but will still generate thumbnails.**