import glob
import os
import torch
import sys

from autofocus.infer import load_model_for_inference, infer, ImageLoader
from lfm_data_utilities.utils import get_corresponding_txt_file
from typing import List, Optional
from zipfile import BadZipFile
from time import perf_counter
from tqdm import tqdm

    
def get_files(data_dir: str) -> List[str]:
    """Get all zarr files in directory"""

    file_format = f'{data_dir}/*-*-*-*_/*.zip'
        
    a = perf_counter()
    files = glob.glob(file_format)
        
    print(files)
    print(f"Number of files: {len(files)}")
        
    b = perf_counter()
    print(f"Get files: {b-a} s")

    return(files)

def load_model(model_dir: str) -> List[str]:
    """Load SSAF or YOGO model"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = load_model_for_inference(model_dir, device)

    return model

def process_files(files: List[str], model: str, output_dir: str, model_type: str) -> None:
    """Get inference values for every frame in each file. Ignore already processed files and bad zipfiles"""

    for file in files:
        basename = pathlib.Path(file).stem
        
        try:
            images = (ImageLoader.load_zarr_data(file))
        except BadZipFile:
            print(f"Skipping {basename}: BadZipFile")
            continue

        output_file = get_corresponding_txt_file(file, output_dir, model_type)
        print(output_file)
        #f"{output_dir}/{basename.removesuffix('.zip')}__{model_type}.txt"

        if os.path.exists(output_file):
            print(f"Skipping {basename}: Already processed")
            continue
            
        print(f"Started processing {basename}")

        c = perf_counter()
        with open(output_file, 'w') as file:
            for res in tqdm(infer(model, images)):
                if model_type == 'ssaf':
                    file.write(f"{res}\n")
                elif model_type == 'yogo':
                    print(res)
        d = perf_counter()
        print(f"Finished writing {basename} data in {d-c} s")

def run(scope_dir: str, model_dir: str, output_dir: str, model_type: str) -> None:
    """Run all the steps to get SSAF data from all zarr files"""

    model = load_model(model_dir)
    files = get_files(scope_dir)
    process_files(files, model, output_dir, model_type)


if __name__ == "__main__":
    try:
        scope_folder = sys.argv[1]
        model_file = sys.argv[2]
        model_type = sys.argv[3]
        output_folder = sys.argv[4]
    except IndexError:
        raise ValueError(
            "Expected format 'python3 infer_rerun_all.py <path to scope folder> <path to model .pth file> <model type> <path to output folder>'"
        )

    valid_types = ['yogo', 'ssaf']
    if not model_type in valid_types:
        raise ValueError(
            "Invalid model type provided. Allowed model types: {valid_types}"
        )

    run(scope_folder, model_file, output_folder, model_type)
    print("Finished processing")
