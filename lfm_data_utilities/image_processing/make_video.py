import sys
from functools import partial
from lfm_data_utilities.utils import *

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <path folders> <path to save videos (note: folder will be created if it doesn't exist already)>")
        sys.exit(1)

    path_to_runset = sys.argv[1]
    path_to_save = sys.argv[2]
    datasets = load_datasets(path_to_runset)
    valid_datasets = [d for d in datasets if d.successfully_loaded]

    make_video_with_path = partial(make_video, save_dir=Path(path_to_save))
    print("Generating videos...")
    with Pool() as pool:
        tqdm(pool.imap(make_video_with_path, valid_datasets), total=len(valid_datasets))