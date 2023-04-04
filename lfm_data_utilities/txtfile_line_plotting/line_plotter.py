import sys
import os

import matplotlib.pyplot as plt
import numpy as np


def plotter(fdatasets, names, ylabel):
    """Plot all datasets on the same graph, with appropriate labels"""
    for dataset, name in zip(datasets, names):
        rms = get_rms(dataset)
        plt.plot(dataset, label="{0}, RMS={1:.3f}]".format(name, rms), alpha=0.4, linewidth=0.5)

    plt.legend()
    plt.title(title)

    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel("Frame")

    plt.show()

def extractor(folder):
    """Find all files in data folder. 

    Returns list of tuples for each file, containing extracted data and dataset name.
    """
    datasets = []
    files = os.listdir(folder)

    for file in files:
        dataset = []

        with open(os.path.join(folder, file), 'r') as f:
            f.readline()
            for line in f:
                dataset.append(float(line.strip()))

        datasets.append(dataset)

    return datasets, files

def get_rms(data):
    """Compute root mean square (rms)"""
    ms = 0
    N = len(data)

    for val in data:
        ms += val**2

    return np.sqrt(ms / N)

def run(folder, title, ylabel=None):
    """Run all the steps to extract and plot the data"""
    datasets, names = extractor(folder)
    plotter(datasets, names, ylabel)

if __name__ == "__main__":
    try:
        folder = sys.argv[1]
    except IndexError:
        raise Exception(
            "Expected format 'python line_plotter.py <dataset folder> [ylabel] [title]'"
        )

    try:
        title = sys.argv[3]
    except IndexError:
        title = folder

    try:
        ylabel = sys.argv[2]
        run(folder, title, ylabel=ylabel)
    except IndexError:
        run(folder, title)