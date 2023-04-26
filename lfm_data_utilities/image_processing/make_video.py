import sys
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path

from lfm_data_utilities import utils

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <path folders> <path to save videos (note: folder will be created if it doesn't exist already)>"
        )
        sys.exit(1)

    path_to_runset = sys.argv[1]
    path_to_save = Path(sys.argv[2])
    datasets = utils.load_datasets(path_to_runset)
    valid_datasets = [d for d in datasets if d.successfully_loaded]

    print("Generating videos...")
    with mp.Pool() as pool:
        pool.starmap(utils.make_video, [(x, path_to_save) for x in tqdm(valid_datasets)])
