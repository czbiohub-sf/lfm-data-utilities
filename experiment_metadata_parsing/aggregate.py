"""
Aggregate analysis of "per_image" metadata files on a per-scope basis.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import *


# for reference
# keys(['im_counter', 'timestamp', 'motor_pos', 'pressure_hpa', 'pressure_status_flag',
# # 'syringe_pos', 'flowrate', 'focus_error', 'temperature', 'humidity', 'looptime',
# # 'runtime', 'zarrwriter_qsize']


def plot(data):
    """Plot flowrate / focus error / syringe position / motor position / total number of images and save the figure."""

    scope_name = input("Enter the scope name: ")

    fig = plt.figure(12, figsize=(10, 7))
    gs = gridspec.GridSpec(3, 4)
    fig.suptitle(f"{scope_name}, num_datasets={len(data)}")

    flowrates: List[List[float]] = []
    focus_errs: List[List[float]] = []
    syringe_pos: List[List[float]] = []
    motor_pos: List[List[float]] = []

    param_data = [flowrates, focus_errs, syringe_pos, motor_pos]

    for i, param in enumerate(["flowrate", "focus_error", "syringe_pos", "motor_pos"]):
        for d in data:
            param_data[i].append([float(x) for x in d[param] if x != ""])

    ax_fr = plt.subplot(gs[0, :2])
    [ax_fr.plot(flowrate, alpha=0.5) for flowrate in flowrates]
    ax_fr.set_title("Flowrate")

    ax_focus = plt.subplot(gs[0, 2:])
    [ax_focus.plot(focus_err, alpha=0.5) for focus_err in focus_errs]
    ax_focus.set_title("Focus error")

    ax_syringe = plt.subplot(gs[1, :2])
    [ax_syringe.plot(sp, alpha=0.5) for sp in syringe_pos]
    ax_syringe.set_title("Syringe position")

    ax_motor = plt.subplot(gs[1, 2:])
    [ax_motor.plot(mp, alpha=0.5) for mp in motor_pos]
    ax_motor.set_title("Motor position")

    ax_run_length = plt.subplot(gs[2, 1:3])
    ax_run_length.bar(range(len(data)), [len(d["im_counter"]) for d in data])
    ax_run_length.set_title("Run length (# of images)")
    ax_run_length.set_xlabel("Dataset idx")

    plt.tight_layout()
    plt.savefig(f"{scope_name}_flow_focus_syringe_motor.png")
    plt.show()


def plot_single_param(param: str, data):
    import numpy as np

    for i, d in enumerate(data):
        motor_pos = [float(x) for x in d["motor_pos"] if x != ""]
        focus_err = np.diff(motor_pos[::30])
        plt.plot(focus_err)
        plt.show(block=False)
        plt.pause(0.01)
        input(f"Dataset {i}, press enter to continue...")


def main():
    tld = input("Enter top level directory: ")
    csv_filepaths = get_list_of_per_image_metadata_files(tld)[:]

    print(f"Loading {len(csv_filepaths)} datasets...")
    data: List[Dict] = []
    data = multiprocess_load_csv(csv_filepaths)

    data = [d["vals"] for d in data if len(d["vals"].keys()) > 0]

    plot(data)


if __name__ == "__main__":
    main()
