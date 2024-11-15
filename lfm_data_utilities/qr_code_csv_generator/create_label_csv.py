"""
Create a csv file for use with creating QR codes on the label printer.

The function takes in a desired string and appends, with an underscore,
numbers from 1 to N at the end.
"""

import argparse
import csv

if __name__ == "__main__":
    # Create an argument parser that takes in a string and an integer
    parser = argparse.ArgumentParser(
        "Create a csv file for use with creating QR codes on the label printer."
    )
    parser.add_argument(
        "barcode_string",
        type=str,
        help="The desired string to be appended with numbers.",
    )
    parser.add_argument(
        "N",
        type=int,
        help="The number of labels to create.",
    )

    args = parser.parse_args()
    barcode_string = args.barcode_string
    N = args.N

    # Create a csv file with the desired string and numbers appended
    # Pad the numbers with zeros to ensure they are all the same length
    # Add the current date to the filename
    with open(f"{barcode_string}_labels_{N}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(1, N + 1):
            writer.writerow([f"{barcode_string}_{str(i).zfill(len(str(N)))}"])
