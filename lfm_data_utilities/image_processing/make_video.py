import sys

from lfm_data_utilities.utils import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path folders>")
        sys.exit(1)

    path_to_runset = sys.argv[1]

    # Get zarr file paths + load 'em in
    print("Getting zarr paths + loading zarr files...")
    zarr_paths = get_list_of_zarr_files(path_to_runset)
    zarr_files = multiprocess_load_zarr(zarr_paths)

    # Get per image csv filepaths + load 'em in
    print("Getting per image csv paths + loading csv files...")
    per_img_csv_paths = get_list_of_per_image_metadata_files(path_to_runset)
    per_img_csvs = multiprocess_load_csv(per_img_csv_paths)

    # Verify that the csv paths and zarr paths match
    for c, z in zip(per_img_csv_paths, zarr_paths):
        print(c, z)