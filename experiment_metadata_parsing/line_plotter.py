import sys
import os

import matplotlib.pyplot as plt
import numpy as np


def plotter(folder, title, ylabel=None):
    datasets, names = extractor(folder)

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
    ms = 0
    N = len(data)

    for val in data:
        ms += val**2

    return np.sqrt(ms / N)


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
        plotter(folder, title, ylabel=ylabel)
    except IndexError:
        plotter(folder, title)