import re
import argparse
import pandas as pd

from os import path, listdir
from lfm_data_utilities.utils import get_list_of_experiment_level_metadata_files

VALID_KEYS = [
    "directory",
    "filename",
    "operator_id",
    "participant_id",
    "flowcell_id",
    "target_flowrate",
    "site",
    "notes",
    "scope",
    "camera",
    "exposure",
    "target_brightness",
    "git_branch",
    "git_commit",
]

DEFAULT_KEYS = ["directory", "notes", "git_branch"]

MAX_COLWIDTH = 50
TXT_FILE = "metadata_compilation.txt"


def run(folder, display_keys=DEFAULT_KEYS):
    # Check that requested keys are valid
    for key in display_keys:
        if key not in VALID_KEYS:
            raise ValueError(
                "Invalid metadata column '" + key + "' requested. "
                f"Valid columns are {VALID_KEYS}"
            )

    exp_files = get_list_of_experiment_level_metadata_files(folder)

    # Track dataframes from all exp metadata file
    df_list = []

    for file in exp_files:
        # Get file location
        fileparts = file.parts
        directory = f"{fileparts[-3]}\\{fileparts[-2]}"
        filename = file.name

        try:
            # Get data from exp metadata file
            single_df = pd.read_csv(file)

            # Store file location
            single_df["directory"] = directory
            single_df["filename"] = filename

            df_list.append(single_df)
        except UnicodeDecodeError:
            print(f"Corrupted file: {directory}\\{filename}")
        except pd.errors.EmptyDataError:
            print(f"Empty file: {directory}\\{filename}")

    master_df = pd.concat(df_list, ignore_index=True)
    master_df = master_df.sort_values(by="directory", ignore_index=True)
    master_df = master_df.fillna("-")

    # Truncate notes columns for readability
    truncated_master_df = master_df.copy()
    truncated_master_df["notes"] = [
        note[: MAX_COLWIDTH - 3] + "..." if len(note) > MAX_COLWIDTH - 3 else note
        for note in master_df["notes"]
    ]

    print("\n" + truncated_master_df[display_keys].to_string())

    with open(TXT_FILE, "w") as writer:
        writer.write(master_df[display_keys].to_string())
    print(f"\nWrote untruncated metadata compilation to {TXT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Metadata compiler",
        description=f"By default, only the columns {DEFAULT_KEYS} are displayed. Additional columns can be viewed using the flags below.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="display all experiment metadata columns",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--include",
        dest="key",
        help=f"include the specified column in the displayed metadata, any of {VALID_KEYS} can be specified",
        action="store",
    )

    parser.add_argument(
        "-f",
        "--folder",
        help="select a custom folder to compile metadata from",
        action="store",
        required=True,
    )
    args = parser.parse_args()

    if args.verbose:
        run(args.folder, display_keys=VALID_KEYS)
    elif args.key is not None:
        custom_keys = DEFAULT_KEYS + [args.key]
        print(custom_keys)
        run(args.folder, display_keys=custom_keys)
    else:
        run(args.folder)
