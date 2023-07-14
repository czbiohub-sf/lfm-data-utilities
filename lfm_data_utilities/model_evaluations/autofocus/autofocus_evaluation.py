#! /usr/bin/env python3

from __future__ import annotations

import os
import heapq
import argparse

from pathlib import Path
from typing import DefaultDict

import torch

import autofocus as af

from tqdm import tqdm
from ruamel.yaml import YAML
from collections import defaultdict
from typing import Tuple, List

from filepath_dataloader import get_dataloader
from lfm_data_utilities import utils


# sometimes bruno doesn't like me plotting, even if I am just
# saving and not displaying any plots - so here is a magic incantation
# to please Bruno the Great
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def write_metadata(
    output_dir: Path,
    autofocus_path_to_pth: List[Path],
):
    autofocus_package_id = utils.try_get_package_version_identifier(af)

    # write all the above to meta.yml in output_dir
    yaml = YAML()
    meta = {
        "autofocus_package_id": autofocus_package_id,
        "autofocus_path_to_pth": [str(p.absolute()) for p in autofocus_path_to_pth],
    }
    with open(output_dir / "meta.yml", "w") as f:
        yaml.dump(meta, f)


def guess_model_name(model_path: Path) -> str:
    # we get the path, not the name of the model
    return f"{model_path.parent.name}/{model_path.name}"


def plot_violin(results, args):
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
            f"{guess_model_name(args.path_to_autofocus_pths.pop())}\n{args.dataset_description_file}"
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

        fig.savefig(f"{args.output_dir / 'autofocus_output.png'}", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create dense data from runs for vetting")
    parser.add_argument(
        "dataset_description_file", type=Path, help="path to dataset_description_file"
    )
    parser.add_argument(
        "path_to_autofocus_pths",
        type=Path,
        nargs="+",
        default=None,
        help="path to autofocus pth file or files (if additional files are given, we use mean loss for ranking",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./ssaf_output",
        help="path to output directory (default ./ssaf_output)",
    )
    parser.add_argument(
        "-N",
        "--N",
        type=int,
        default=32,
        help="number of best and worst images (by loss) that will be saved (default 32)",
    )

    args = parser.parse_args()

    if not args.dataset_description_file.exists():
        raise ValueError(f"{args.dataset_description_file} does not exist")
    elif not all(p.exists() for p in args.path_to_autofocus_pths):
        raise ValueError(f"{args.path_to_autofocus_pth} does not exist")

    args.output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for pth_path in args.path_to_autofocus_pths:
        net = af.model.AutoFocus.from_pth(pth_path)
        net.eval()
        net.to(device)
        net = torch.jit.script(net)
        models.append(net)

    dataloaders = get_dataloader(
        args.dataset_description_file,
        img_size=(300, 400),
        batch_size=32,
        split_fractions_override={"eval": 1.0},
        augmentation_split_fraction_name="",
    )

    loss_fn = torch.nn.functional.mse_loss

    # Could make these `heapq` heaps for performance, but then we have to implement the maxsize
    min_pq: List[Tuple[float, float, float, torch.Tensor, str]] = []
    max_pq: List[Tuple[float, float, float, torch.Tensor, str]] = []
    all_losses: List[float] = []

    results: DefaultDict[int, list] = defaultdict(list)
    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloaders["eval"]):
            imgs = imgs.to(device, dtype=torch.float, non_blocking=True)
            labels = labels.to(device, dtype=torch.float, non_blocking=True).view(-1, 1)

            losses = []
            predictions = []
            for model in models:
                preds = model(imgs).view(-1, 1)
                loss = loss_fn(preds, labels, reduction="none")
                predictions.append(preds)
                losses.append(loss)

            # get mean loss across each prediction
            stacked_preds = torch.cat(predictions, dim=1)
            mean_loss = torch.mean(torch.cat(losses, dim=1), dim=1)

            preds = preds.cpu()
            labels = labels.cpu()
            imgs = imgs.cpu()
            for l, pred, label, img, path in zip(
                mean_loss, stacked_preds, labels, imgs, paths
            ):
                all_losses.append(l.item())

                if len(min_pq) < args.N:
                    heapq.heappush(min_pq, (-l, pred, label, img, path))
                else:
                    heapq.heapreplace(min_pq, (-l, pred, label, img, path))

                if len(max_pq) < args.N:
                    heapq.heappush(max_pq, (l, pred, label, img, path))
                else:
                    heapq.heapreplace(max_pq, (l, pred, label, img, path))

            for i, label in enumerate(labels):
                results[label.item()].append(preds[i])

    # plot results
    import matplotlib.pyplot as plt
    import numpy as np

    ii = 0
    for el in tqdm(max_pq, desc="max pq plotting"):
        loss, pred, label, img, path = el
        # fix -los setting for max heap
        plt.imshow(img[0, ...].numpy(), cmap="gray")
        plt.title(
            f"loss {loss.item():.3f}, pred {[round(n.item(),3) for n in pred]}, lbl {label.item():.3f}\n{'/'.join(Path(path).parts[-5:])}",
            fontsize=7,
        )
        plt.savefig(
            f"{(args.output_dir / ('max_' + str(ii))).with_suffix('.png')}", dpi=150
        )
        plt.clf()
        plt.cla()
        ii += 1

    ii = 0
    for el in tqdm(min_pq, desc="min pq plotting"):
        loss, pred, label, img, path = el
        loss = -loss
        plt.imshow(img[0, ...].numpy(), cmap="gray")
        plt.title(
            f"loss {loss.item():.3f}, pred {[round(n.item(),3) for n in pred]}, lbl {label.item():.3f}\n{'/'.join(Path(path).parts[-5:])}",
            fontsize=7,
        )
        plt.savefig(
            f"{(args.output_dir / ('min_' + str(ii))).with_suffix('.png')}", dpi=150
        )
        plt.clf()
        plt.cla()
        ii += 1

    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 12))
    ax.hist(all_losses, bins=50, log=True)
    ax.set_title("loss histogram")
    ax.set_xlabel("loss")
    ax.set_ylabel("Frequency")
    plt.savefig(f"{args.output_dir / 'loss_hist.png'}", dpi=150)

    write_metadata(
        args.output_dir,
        args.path_to_autofocus_pths,
    )

    if len(args.path_to_autofocus_pths) == 1:
        plot_violin(results, args)

    # it takes maybe 15 seconds to shut down dataloader workers,
    # so just let the user know
    print("shutting down")
