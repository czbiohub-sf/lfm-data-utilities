import glob
import os
import torch
import sys

from time import perf_counter
from tqdm import tqdm


class Rerunner():
    def __init__(self, infer, model_loader, image_loader):
        self.infer = infer
        self.model_loader = model_loader
        self.image_loader = image_loader

        self.file_format = f'{data_dir}/*-*-*-*/*-*-*-*_/*.zip'
    
    def get_files(self, data_dir):
        a = perf_counter()
        files = glob.glob(self.file_format)
        
        print(files)
        print(f"Number of files: {len(files)}")
        
        b = perf_counter()
        print(f"Get files: {b-a} s")

        return(files)

    def load_model(self, model_dir):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        model = self.model_loader(model_dir, device)

        return model

    def process_files(self, files, model, output_dir):
        for file in files:
            images = (self.image_loader.load_zarr_data(file))
            basename = os.path.basename(file)
            output_file = f"{output_dir}/{basename.removesuffix('.zip')}__ssaf.txt"

            if output_file.is_file():
                print(f"Skipping {basename}, already processed")
            else:
                print(f"Started processing {basename}")

                c = perf_counter()
                try:
                    with open(output_file, 'w') as file:
                        for res in tqdm(infer(model, images)):
                            file.write(f"{res}\n")
                except BadZipFile:
                    print(f"BadZipFile: {basename}")
                else:
                    d = perf_counter()
                    print(f"Finished writing {basename} SSAF data in {d-c} s")
                    

    def rerun(self, scope_dir, model_dir, output_dir):
        model = load_model(model_dir)
        files = get_files(scope_dir)
        process_files(files, model, output_dir)


if __name__ == "__main__":

    print(sys.argv)
    try:
        ssaf_repo = sys.argv[1]
        scope_folder = sys.argv[2]
        model_file = sys.argv[3]
        output_folder = sys.argv[4]
    except IndexError:
        raise Exception(
            "Expected format 'python3 ssaf_rerun_all.py <path to ulc-malaria-autofocus repo> <path to scope folder> <path to model .pth file> <path to output folder>'"
        )

    try: 
        sys.path.append(os.path.abspath(ssaf_repo))
        from infer import load_model_for_inference, infer, ImageLoader
    except Exception as e:
        print(e)
    
    SSAF_rerunner = Rerunner(load_model_for_inference, infer, ImageLoader)
    SSAF_rerunner.rerun(scope_folder, model_file, output_folder)
