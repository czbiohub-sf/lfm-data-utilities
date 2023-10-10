import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from pathlib import Path


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


if __name__ == "__main__":
    base_dir = Path(
        "/hpc/projects/group.bioengineering/LFM_scope/misc/yogo-statistics"
    )
    folder = Path(sys.argv[1])
    data_file = base_dir / folder / "res.csv"
    sigmoid_file = base_dir / folder / "sigmoid.png"
    bins_file = base_dir / folder / "bins.png"

    all_res = np.loadtxt(data_file, delimiter=",").T
    num_res = np.shape(all_res)[1]

    popt, pcov = curve_fit(sigmoid, all_res[0], all_res[1])
    x = np.linspace(0, 1, num=101)
    y = sigmoid(x, popt[0], popt[1])

    bin_count = 20
    bins = np.linspace(0, 1, num=bin_count + 1)
    bin_avg = np.empty(bin_count)
    bin_centers = np.empty(bin_count)
    for idx in range(bin_count):
        bin_filt = np.logical_and(all_res[0] > bins[idx], all_res[0] < bins[idx + 1])
        bin_avg[idx] = np.mean(all_res[1][bin_filt])
        bin_centers[idx] = (bins[idx] + bins[idx + 1]) / 2.0

    plt.scatter(all_res[0], all_res[1], label="Binarized result")
    plt.plot(x, y, color="orange", label="Sigmoid fit")
    plt.title(f"A={popt[0]:.2f}, B={popt[1]:.2f}, N={num_res}")
    plt.legend()
    plt.savefig(sigmoid_file)

    plt.figure()
    plt.scatter(all_res[0], all_res[1], label="Binarized result")
    plt.plot(bin_centers, bin_avg, color="red", label="Binned average")
    plt.title(f"Bins={bin_count}, N={num_res}")
    plt.legend()
    plt.savefig(bins_file)

    print(f"Saved data to {base_dir / folder}")
