import sys
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np


def plotter(datasets, names, ylabel, legend, title, output):
    """Plot all datasets on the same graph, with appropriate labels"""
    for dataset, name in zip(datasets, names):
        if legend:
            rms = get_rms(dataset)
            plt.plot(dataset, label="{0}, RMS={1:.3f}]".format(name, rms), alpha=0.4, linewidth=0.5)
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
    datasets = []
    files = os.listdir(folder)

    if len(files) > 10:
        legend = False
    else:
        legend = True

    for file in files:
        dataset = []

        with open(os.path.join(folder, file), 'r') as f:
            print("READ")
            try:
                f.readlines()
            except UnicodeDecodeError:
                print(f"Skipping corrupted file: {file}")
                continue
                
            for line in f:
                dataset.append(float(line.strip()))

        print("UNSKIPPED")
        datasets.append(dataset)

    return datasets, files, legend

def get_rms(data):
    """Compute root mean square (rms)"""
    ms = 0
    N = len(data)

    for val in data:
        ms += val**2

    return np.sqrt(ms / N)

def run(folder, title, ylabel, output=None):
    """Run all the steps to extract and plot the data"""
    datasets, names, legend = extractor(folder)
    plotter(datasets, names, ylabel, legend, output)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--dir", help="Directory containing txtfile data folder", required=True)
    argparser.add_argument("-y", "--ylabel", help="Label for y-axis", required=True)
    argparser.add_argument("-t", "--title", help="Title for plot")
    argparser.add_argument("-o", "--output", help="Filename to export plot to")

    args = parser.parse_args()

    if args.title:
        title = args.title
    else:
        title = folder

    if args.output:
        run(folder, title, ylabel, args.output)
    else:
        run(folder, title, ylabel, args.output)
