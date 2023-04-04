import glob
import os
import torch
import sys

from autofocus.infer import load_model_for_inference, infer, ImageLoader
from zipfile import BadZipFile
from time import perf_counter
from tqdm import tqdm

    
def get_files(data_dir):
    """Get all zarr files in directory"""
    file_format = f'{data_dir}/*-*-*-*_/*.zip'
        
    a = perf_counter()
    files = glob.glob(file_format)
        
    print(files)
    print(f"Number of files: {len(files)}")
        
    b = perf_counter()
    print(f"Get files: {b-a} s")

    return(files)

def load_model(model_dir):
    """Load SSAF model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = load_model_for_inference(model_dir, device)

    return model

def process_files(files, model, output_dir):
    """Get SSAF values for every frame in each file. Ignore already processed files and bad zipfiles"""
    for file in files:
        basename = pathlib.Path(file).stem
        
        try:
            images = (ImageLoader.load_zarr_data(file))
        except BadZipFile:
            print(f"Skipping {basename}: BadZipFile")
            continue

        output_file = f"{output_dir}/{basename.removesuffix('.zip')}__ssaf.txt"

        if os.path.exists(output_file):
            print(f"Skipping {basename}: Already processed")
            continue
            
        print(f"Started processing {basename}")

        c = perf_counter()
        with open(output_file, 'w') as file:
            for res in tqdm(infer(model, images)):
                file.write(f"{res}\n")
        d = perf_counter()
        print(f"Finished writing {basename} SSAF data in {d-c} s")

def run(scope_dir, model_dir, output_dir):
    """Run all the steps to get SSAF data from all zarr files"""
    model = load_model(model_dir)
    files = get_files(scope_dir)
    process_files(files, model, output_dir)


if __name__ == "__main__":
    try:
        scope_folder = sys.argv[1]
        model_file = sys.argv[2]
        output_folder = sys.argv[3]
    except IndexError:
        raise Exception(
            "Expected format 'python3 ssaf_rerun_all.py <path to scope folder> <path to model .pth file> <path to output folder>'"
        )

    run(scope_folder, model_file, output_folder)
    print("Finished processing")
