import os
import glob
from collections import defaultdict


def count_labels(base_path):
    label_counts = defaultdict(
        lambda: defaultdict(int)
    )  # Nested dictionaries for counts
    # Search for all 'labels_plus' directories within the base_path
    for labels_dir in glob.glob(
        os.path.join(base_path, "**/labels_plus"), recursive=True
    ):
        # Iterate over each file in the found directory
        for filename in glob.glob(os.path.join(labels_dir, "*.txt")):
            with open(filename, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 6:  # Ensure the line has exactly 6 values
                        class_id = parts[0]
                        verified = "Human" if parts[-1] == "1" else "Machine"
                        label_counts[class_id][verified] += 1
    return label_counts


# Usage
base_directory = (
    "/hpc/projects/group.bioengineering/LFM_scope/merged_labels/still-voice-4405"
)
result = count_labels(base_directory)
for class_id, counts in sorted(result.items()):
    print(f"Class {class_id}:")
    for type_label, count in sorted(counts.items()):
        print(f"  {type_label} : {count:,}")