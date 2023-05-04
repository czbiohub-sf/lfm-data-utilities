#! /usr/bin/env python3

import os
import argparse

from pathlib import Path
from typing import DefaultDict

import torch

import autofocus as af

from tqdm import tqdm
from ruamel.yaml import YAML
from collections import defaultdict

from lfm_data_utilities import utils


# sometimes bruno doesn't like me plotting, even if I am just
# saving and not displaying any plots - so here is a magic incantation
# to please Bruno the Great
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def write_metadata(
    output_dir: Path,
    autofocus_path_to_pth: Path,
):
    autofocus_package_id = utils.try_get_package_version_identifier(af)

    # write all the above to meta.yml in output_dir
    yaml = YAML()
    meta = {
        "autofocus_package_id": autofocus_package_id,
        "autofocus_path_to_pth": str(autofocus_path_to_pth.absolute()),
    }
    with open(output_dir / "meta.yml", "w") as f:
        yaml.dump(meta, f)


def guess_model_name(model_path: Path) -> str:
    # we get the path, not the name of the model
    return f"{model_path.parent.name}/{model_path.name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create dense data from runs for vetting")
    parser.add_argument(
        "dataset_description_file", type=Path, help="path to dataset_description_file"
    )
    parser.add_argument(
        "path_to_autofocus_pth",
        type=Path,
        default=None,
        help="path to autofocus pth file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./ssaf_output",
        help="path to output directory (default ./ssaf_output)",
    )

    args = parser.parse_args()

    if not args.dataset_description_file.exists():
        raise ValueError(f"{args.dataset_description_file} does not exist")
    elif not args.path_to_autofocus_pth.exists():
        raise ValueError(f"{args.path_to_autofocus_pth} does not exist")

    args.output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = af.model.AutoFocus.from_pth(args.path_to_autofocus_pth)
    net.eval()
    net.to(device)
    net = torch.jit.script(net)

    dataloaders = af.dataloader.get_dataloader(
        args.dataset_description_file,
        img_size=(300, 400),
        batch_size=32,
        split_fractions_override={"eval": 1.0},
        augmentation_split_fraction_name="",
    )

    results: DefaultDict[int, list] = defaultdict(list)
    with torch.no_grad():
        for imgs, labels in tqdm(dataloaders["eval"]):
            imgs = imgs.to(device, dtype=torch.float, non_blocking=True)
            labels = labels.to(device, dtype=torch.float, non_blocking=True)
            preds = net(imgs).view(-1)

            for i, label in enumerate(labels):
                results[label.item()].append(preds[i].item())

    # plot results
    import matplotlib.pyplot as plt
    import numpy as np

    write_metadata(
        args.output_dir,
        args.path_to_autofocus_pth,
    )

    def set_violin_plot_color(parts):
        for pc in parts["bodies"]:
            pc.set_facecolor("#D43F3A")
            pc.set_edgecolor("black")
            pc.set_alpha(1)

    mini, maxi = min(results.keys()), max(results.keys())
    with utils.timing_context_manager("plotting"):
        fig, (whole_range_ax, tight_range_ax) = plt.subplots(1, 2, figsize=(16, 12))
        whole_range_ax.set_facecolor((0.95, 0.95, 0.95))
        fig.suptitle(
            f"{guess_model_name(args.path_to_autofocus_pth)}\n{args.dataset_description_file}"
        )
        whole_range_ax.plot(
            [mini, maxi],
            [mini, maxi],
            linestyle="--",
            color="gray",
            linewidth=1,
            alpha=0.5,
        )
        tight_range_ax.plot(
            [-10, 10],
            [-10, 10],
            linestyle="--",
            color="gray",
            linewidth=1,
            alpha=0.5,
        )
        # plot a dashed line in gray at y=x
        for label, values in results.items():
            # plot candle plots for each label
            npvalues = np.array(values)
            vps = whole_range_ax.violinplot(
                npvalues,
                positions=[label],
                widths=0.7,
                showmeans=True,
                showextrema=False,
                showmedians=False,
            )
            set_violin_plot_color(vps)
            if abs(label) <= 10:
                vps = tight_range_ax.violinplot(
                    npvalues,
                    positions=[label],
                    widths=0.4,
                    showmeans=True,
                    showextrema=False,
                    showmedians=False,
                )
                set_violin_plot_color(vps)

        whole_range_ax.set_xlabel("label")
        whole_range_ax.set_ylabel("autofocus output")
        tight_range_ax.set_xlabel("label")
        tight_range_ax.set_ylabel("autofocus output")

        fig.savefig(f"{args.output_dir / 'autofocus_output.png'}", dpi=300)

    # it takes maybe 15 seconds to shut down dataloader workers,
    # so just let the user know
    print("shutting down")
