import argparse
from collections import defaultdict
import glob
import os
from pathlib import Path

from lfm_data_utilities import YOGO_CLASS_ORDERING


def count_labels(base_path):
    label_counts = defaultdict(
        lambda: defaultdict(int)
    )  # Nested dictionaries for counts
    # Search for all 'labels_plus' directories within the base_path
    path = os.path.join(base_path, "**/labels_plus")
    if not Path(path).exists():
        # Let's use the regular labels
        path = os.path.join(base_path, "**/labels")

    for labels_dir in glob.glob(path, recursive=True):
        # Iterate over each file in the found directory
        for filename in glob.glob(os.path.join(labels_dir, "*.txt")):
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


if __name__ == "__main__":
    # Use argparse to get the path
    parser = argparse.ArgumentParser(
        description="This script processes a directory structure to count labeled data, distinguishing between 'Human' verified and 'Machine' generated labels. It requires the path to the base directory containing 'labels_plus' subdirectories."
    )
    parser.add_argument("base_path", help="Path to the base directory")
    args = parser.parse_args()

    base_directory = args.base_path
    result = count_labels(base_directory)
    for class_id, counts in sorted(result.items()):
        print(f"Class {YOGO_CLASS_ORDERING[int(class_id)]}:")
        for type_label, count in sorted(counts.items()):
            print(f"  {type_label} : {count:,}")
