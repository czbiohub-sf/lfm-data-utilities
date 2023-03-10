import numpy as np
import matplotlib.pyplot as plt

import allantools
import sys


def super_plotter(data, title, throttle=60, output=None):
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2,3)
    ax_allan = fig.add_subplot(gs[0, 0])
    ax_data = fig.add_subplot(gs[1, 0])
    ax_throttled_allan = fig.add_subplot(gs[0, 1])
    ax_throttled_data = fig.add_subplot(gs[1, 1])
    ax_halflife = fig.add_subplot(gs[:, 2])

    alphas = [0.01, 0.03, 0.05, 0.1]
    throttled_data = data[::int(throttle)]

    data_plotter(data, ax_allan, ax_data, 1)
    data_plotter(data, ax_throttled_allan, ax_throttled_data, int(throttle))

    print(np.linspace(0.1, 1, 9))
    halflife_plotter(np.linspace(0.1, 0.9, 9), ax_halflife)

    fig.suptitle(title)
    fig.tight_layout()

    if output is not None:
        fig.savefig(output)

    plt.show()


def data_plotter(raw_data, ax_allan, ax_data, throttle):

    data = raw_data[::throttle]
    frames = np.arange(0, len(data))

    if throttle > 1:
        xlabel = "Throttled frame(s)"
        ax_allan.set_title("Unthrottled [STD = {0:.3f}]".format(np.std(data)))
        frames = frames * throttle
    else:
        xlabel = "Frame(s)"
        ax_allan.set_title("{0} frame throttle [STD = {1:.3f}]".format(throttle, np.std(data)))

    t = np.logspace(0, 3, 50)  # tau values from 1 to 1000
    t2, ad, _, _ = allantools.tdev(data, taus=t)
    ax_allan.loglog(t2*throttle, ad, label="Unfiltered")
    ax_data.plot(frames, data, label="Unfiltered", alpha=0.5)

    for alpha in [0.01, 0.03, 0.05, 0.1]:
        ewma_vals = []
        prev_data = data[0]
        for j in data[1:]:
            prev_data = prev_data * (1 - alpha) + alpha * j
            ewma_vals.append(prev_data)
        t2, ad, _, _ = allantools.tdev(ewma_vals, taus=t)
        ax_allan.loglog(t2*throttle, ad, label=f"alpha={alpha}")
        ax_data.plot(frames[1:], ewma_vals, label=f"alpha={alpha}", alpha=0.5)

    ax_allan.set_ylabel("Allan deviation")
    ax_allan.set_xlabel(xlabel)
    ax_allan.legend()

    ax_data.set_ylim(-4, 28)
    ax_data.set_ylabel("SSAF error [steps]")
    ax_data.set_xlabel(xlabel)
    ax_data.legend()

def halflife_plotter(alphas, ax):
    adjustment_periods = []

    for alpha in alphas:
        halflife = -np.log(2) / np.log(1 - alpha)
        adjustment_periods.append(2 * halflife)

    ax.plot(alphas, adjustment_periods)
    ax.set_ylabel("Adjustment period [frames]")
    ax.set_xlabel("Alpha")
    ax.set_title("Adjustment period vs. alpha")



if __name__ == "__main__":

    num_inputs = len(sys.argv)

    if num_inputs == 1:
        raise ValueError(
            "Expected a txt target file containing all datapoints delimited by newline.\n"
            "Command format: <txt data file> [throttle number] [title] [output file]"
            )
    elif num_inputs > 5:
        raise ValueError(
            "Too many arguments"
            "Command format: <txt data file> [throttle number] [title] [output file]"
            )
    
    filename = sys.argv[1]
    data = np.genfromtxt(filename, delimiter='\n')

    if num_inputs == 2:
        super_plotter(data, filename)
    elif num_inputs == 3:
        super_plotter(data, filename, throttle=sys.argv[2])
    elif num_inputs == 4:
        super_plotter(data, sys.argv[3], throttle=sys.argv[2])
    elif num_inputs == 5:
        super_plotter(data, sys.argv[3], throttle=sys.argv[2], output=sys.argv[4])