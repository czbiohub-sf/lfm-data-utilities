import numpy as np
import matplotlib.pyplot as plt

import allantools
import sys


def ewma_allan_dev(data, title, output=None):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
    ax_allan = ax[0]
    ax_data = ax[1]

    x_vals = np.arange(1,)

    t = np.logspace(0, 3, 50)  # tau values from 1 to 1000
    t2, ad, _, _ = allantools.tdev(data, taus=t)
    ax_allan.loglog(t2, ad, label="Unfiltered")
    ax_data.plot(data, label="Unfiltered", alpha=0.5)

    for alpha in [0.01, 0.05, 0.1]:
        ewma_vals = []
        prev_data = data[0]
        for j in data[1:]:
            prev_data = prev_data * (1 - alpha) + alpha * j
            ewma_vals.append(prev_data)
        t2, ad, _, _ = allantools.tdev(ewma_vals, taus=t)
        ax_allan.loglog(t2, ad, label=f"alpha={alpha}")
        ax_data.plot(ewma_vals, label=f"alpha={alpha}", alpha=0.5)

    ax_allan.set_ylabel("Allan deviation")
    ax_allan.set_xlabel("Frame(s)")
    ax_allan.legend()

    ax_data.set_ylim(-4, 28)
    ax_data.set_ylabel("SSAF error [steps]")
    ax_data.set_xlabel("Frame(s)")
    ax_data.legend()

    plt.suptitle(title + " [STD = {0:.3f}]".format(np.std(data)))

    if output is not None:
        plt.savefig(output)

    plt.show()


if __name__ == "__main__":

    num_inputs = len(sys.argv)

    if num_inputs == 1:
        raise ValueError("Expected a txt target file containing all datapoints delimited by newline.")
    elif num_inputs > 4:
        raise ValueError("Too many arguments")

    filename = sys.argv[1]
    data = np.genfromtxt(filename, delimiter='\n')

    if num_inputs == 2:
        ewma_allan_dev(data, filename)
    elif num_inputs == 3:
        ewma_allan_dev(data, sys.argv[2])
    elif num_inputs == 4:
        ewma_allan_dev(data, sys.argv[2], sys.argv[3])