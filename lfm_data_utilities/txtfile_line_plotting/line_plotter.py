import os
import argparse

import matplotlib.pyplot as plt

from lfm_data_utilities.utils import load_txtfile, get_rms


def plotter(datasets, names, ylabel, legend, title, output):
    """Plot all datasets on the same graph, with appropriate labels"""
    for dataset, name in zip(datasets, names):
        if legend:
            rms = get_rms(dataset)
            plt.plot(
                dataset,
                label="{0}, RMS={1:.3f}]".format(name, rms),
                alpha=0.4,
                linewidth=0.5,
            )
        else:
            plt.plot(dataset, alpha=0.4, linewidth=0.5)

    if legend:
        plt.legend()

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Frame")

    if output:
        plt.savefig(output)
        print(f"saved image to {output}")
    plt.show()


def extractor(folder):
    """Find all files in data folder.

    Returns list of tuples for each file, containing extracted data and dataset name.
    """
    files = os.listdir(folder)

    if len(files) > 10:
        legend = False
    else:
        legend = True

    datasets = [load_txtfile(os.path.join(folder, file)) for file in files]

    return datasets, files, legend


def run(folder, title, ylabel, output=None):
    """Run all the steps to extract and plot the data"""
    datasets, names, legend = extractor(folder)

    output_file = os.path.join(folder, f"{output}.png")
    plotter(datasets, names, ylabel, legend, title, output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d", "--dir", help="Directory containing txtfile data folder", required=True
    )
    argparser.add_argument("-y", "--ylabel", help="Label for y-axis", required=True)
    argparser.add_argument("-t", "--title", help="Title for plot")
    argparser.add_argument("-o", "--output", help="Filename to export plot to")

    args = argparser.parse_args()

    if args.title:
        title = args.title
    else:
        title = args.dir

    if args.output:
        run(args.dir, title, args.ylabel, output=args.output)
    else:
        run(args.dir, title, args.ylabel)
