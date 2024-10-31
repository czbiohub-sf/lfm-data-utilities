import argparse
from collections import defaultdict
import concurrent.futures
import os
from pathlib import Path
import yaml

from tqdm import tqdm

from lfm_data_utilities import YOGO_CLASS_ORDERING


def count_labels(path):
    label_counts = defaultdict(
        lambda: defaultdict(int)
    )  # Nested dictionaries for counts

    # Check if a labels_plus directory exists
    if (Path(path).parent / "labels_plus").exists():
        path = Path(path).parent / "labels_plus"

    # Iterate over each file in the found directory
    files = [os.path.join(path, x) for x in os.listdir(path) if ".txt" in x]
    for filename in files:
        with open(filename, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 6:  # Ensure the line has exactly 6 values
                    class_id = parts[0]
                    verified = "Human" if parts[-1] == "1" else "Machine"
                    label_counts[class_id][verified] += 1
                elif len(parts) == 5:  # Of unknown provenance
                    class_id = parts[0]
                    verified = "Unknown"
                    label_counts[class_id][verified] += 1
    return label_counts


def process_label_path(label_path):
    return count_labels(label_path)


if __name__ == "__main__":
    # Use argparse to get the path
    parser = argparse.ArgumentParser(
        description="This script processes a dataset definition file to count labeled data, distinguishing between 'Human' verified, 'Machine' generated, and 'Unknown' labels \
            (unknown labels are those which do not have a column distinguishing whether it was human verified or machine generated). It requires the path to the base directory containing 'labels_plus' subdirectories."
    )
    parser.add_argument("base_path", help="Path to the dataset definition YAML file.")
    args = parser.parse_args()

    # Check if it ends with a `.yml` extension
    if not args.base_path.endswith(".yml"):
        raise ValueError("The file must be a YAML file.")

    with open(args.base_path, "r") as file:
        vals = yaml.safe_load(file)
        label_paths = []
        for x in vals["dataset_paths"].keys():
            label_paths.append(Path(vals["dataset_paths"][x]["label_path"]))

    # Loop through the parent parent_label_paths, get counts and concatenate them all
    result = defaultdict(lambda: defaultdict(int))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_label_path, label_path): label_path
            for label_path in label_paths
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            counts = future.result()
            for class_id, counts in counts.items():
                for verified, count in counts.items():
                    result[class_id][verified] += count

    # Sort based on YOGO_CLASS_ORDERING and output the class name
    sorted_result = sorted(
        result.items(),
        key=lambda x: YOGO_CLASS_ORDERING.index(YOGO_CLASS_ORDERING[int(x[0])]),
    )

    # Print the result
    for class_id, counts in sorted_result:
        print(f"Class {YOGO_CLASS_ORDERING[int(class_id)]}:")
        for type_label, count in sorted(counts.items()):
            print(f"  {type_label} : {count:,}")
