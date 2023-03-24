import sys

from lfm_data_utilities.utils import *

def open_zarr_or_return_flag(zarr_path: str):
    try:
        return zarr.open(zarr_path, "r")
    except:
        return "ERR"
    
def custom_multiprocess_load_zarr(zarr_paths: List[Path]):
    return multiprocess_load_files(zarr_paths, open_zarr_or_return_flag)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path folders>")
        sys.exit(1)

    path_to_runset = sys.argv[1]

    valid_datasets = get_full_datasets(path_to_runset)
    print(valid_datasets)