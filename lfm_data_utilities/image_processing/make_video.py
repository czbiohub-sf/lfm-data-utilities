import sys

from lfm_data_utilities.utils import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path folders>")
        sys.exit(1)

    path_to_runset = sys.argv[1]
    valid_datasets = get_valid_datasets(get_all_datasets(path_to_runset))
    zarr_files = multiprocess_load_zarr([d.zarr_path for d in valid_datasets])
    per_img_csv_files = multiprocess_load_csv([d.per_img_csv_path for d in valid_datasets])

    for d in valid_datasets:
        print(d.zarr_path.parent == d.per_img_csv_path.parent)
