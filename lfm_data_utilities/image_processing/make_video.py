import sys

from lfm_data_utilities.utils import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path folders>")
        sys.exit(1)

    path_to_runset = sys.argv[1]
    datasets = load_datasets(path_to_runset)
    valid_datasets = [d for d in datasets if d.successfully_loaded]
