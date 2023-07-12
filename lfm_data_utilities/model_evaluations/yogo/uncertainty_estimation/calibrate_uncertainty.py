import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from pathlib import Path

from yogo.data import YOGO_CLASS_ORDERING


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))


if __name__ == '__main__':
    base_dir = Path("/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/misc/yogo-statistics")
    folder = Path(sys.argv[1])
    data_file = base_dir / folder / "res.csv"
    bins_file = base_dir / folder / "bins.png"
    
    all_res = np.loadtxt(data_file, delimiter=',').T
    
    for class_idx, class_name in enumerate(YOGO_CLASS_ORDERING):
        class_filter = all_res[2] == class_idx
        class_res = all_res[0:2, class_filter]

        num_res = np.shape(class_res)[1]

        popt, pcov = curve_fit(sigmoid, class_res[0], class_res[1])
        x = np.linspace(0, 1, num=101)
        y = sigmoid(x, popt[0], popt[1])

        num_bins = 20
        bins = np.linspace(0, 1, num=num_bins+1)
        bin_avg = np.empty(num_bins)
        bin_centers = np.empty(num_bins)
        for idx in range(num_bins):
            bin_filt = np.logical_and(class_res[0] > bins[idx], class_res[0] < bins[idx+1])
            bin_avg[idx] = np.mean(class_res[1][bin_filt])
            bin_centers[idx] = (bins[idx] + bins[idx+1]) / 2.0

        plt.figure()
        plt.scatter(class_res[0], class_res[1], label="Binarized result", s=5)
        plt.plot(x, y, color="orange", label="Sigmoid fit")
        plt.title(f"{class_name.upper()}: sig_A={popt[0]:.2f}, sig_B={popt[1]:.2f}, bins={num_bins},  N={num_res}")
        plt.plot(bin_centers, bin_avg, color='red', label="Binned average")
        plt.ylabel("Correctness")
        plt.xlabel("Confidence")
        plt.legend()
        plt.savefig(base_dir / folder / f"{class_name.lower()}.png")
       
    print(f"Saved data to {base_dir / folder}")
