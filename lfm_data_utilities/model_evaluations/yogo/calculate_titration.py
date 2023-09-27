#! /usr/bin/env python3

"""
Gargantuan, gross script to calculate titration curves for a given model
"""


import csv
import math
import warnings
import argparse

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError

import matplotlib.pyplot as plt
import numpy as np
from ruamel import yaml
import torch

from yogo.infer import (
    predict,
    count_cells_for_formatted_preds,
)
from yogo.utils import format_preds
from yogo.data import YOGO_CLASS_ORDERING

from lfm_data_utilities import utils


def load_titration_yml(
    path_to_titration_yml: Path,
) -> Tuple[Dict[str, Path], float, Dict[str, Path]]:
    """
    Returns
    -------
    Tuple of the following:
        Dict[str, Path]
            Path to each titration point's images folder
        float
            Starting parasitemia
        Dict[str, Path]
            Paths to the heatmap_masks (will return mappings to None objects if the heatmap_masks key isn't provided in the yaml file)
    """

    point_to_path: Dict[str, Path] = dict()
    point_to_heatmap_mask_path: Dict[str, Path] = dict()

    with open(path_to_titration_yml, "r") as f:
        yaml_data = yaml.safe_load(f)
        points: Dict[str, str] = yaml_data["titration-points"]
        initial_parasitemia: float = float(yaml_data["initial-titration-parasitemia"])
        if not (0 <= initial_parasitemia <= 1):
            raise ValueError(
                f"initial_parasitemia must be between 0 and 1; got {initial_parasitemia}"
            )

        for titration_point, path in points.items():
            tpoint_path = Path(path)
            if not tpoint_path.exists():
                raise ValueError(
                    f"Path {path} for titration point {titration_point} does not exist"
                )
            point_to_path[titration_point] = tpoint_path

        if "heatmap_masks" in yaml_data.keys():
            print("Loading in heatmap masks...")
            heatmap_mask_points: Dict[str, str] = yaml_data["heatmap_masks"]
            for titration_point, path in heatmap_mask_points.items():
                heatmap_mask_path = Path(path)
                if not heatmap_mask_path.exists():
                    raise ValueError(
                        f"'heatmap_masks' is present as a key in the yaml file, but the path doesn't seem to exist: {heatmap_mask_path}"
                    )
                point_to_heatmap_mask_path[titration_point] = heatmap_mask_path
        else:
            for titration_point, _ in points.items():
                point_to_heatmap_mask_path[titration_point] = None

    return point_to_path, initial_parasitemia, point_to_heatmap_mask_path


def process_prediction(
    predictions: torch.Tensor,
    titration_point: str,
    result_dict: Dict[
        str, Dict[str, Union[List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]]
    ],
    min_confidence_threshold: Optional[float] = None,
    heatmap_mask: Optional[np.ndarray] = None,
) -> None:
    per_image_counts: List[torch.Tensor] = []

    confidence_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    confidence_range_sums = {
        cv: torch.zeros(len(YOGO_CLASS_ORDERING), dtype=torch.long)
        for cv in confidence_values
    }

    tot_class_sum = torch.zeros(len(YOGO_CLASS_ORDERING), dtype=torch.long)

    for pred_slice in predictions:
        pred = format_preds(pred_slice, heatmap_mask=heatmap_mask)

        if pred.numel() == 0:
            continue  # ignore no predictions

        classes = pred[:, 5:]

        for confidence_threshold in confidence_values:
            image_counts = count_cells_for_formatted_preds(
                classes, min_confidence_threshold=confidence_threshold
            )
            confidence_range_sums[confidence_threshold] += image_counts

        image_counts = count_cells_for_formatted_preds(
            classes, min_confidence_threshold=min_confidence_threshold
        )
        tot_class_sum += image_counts

        per_image_counts.append(
            tot_class_sum / tot_class_sum.sum()
            if tot_class_sum.sum() > 0
            else torch.zeros(len(YOGO_CLASS_ORDERING))
        )

    result_dict[titration_point] = {
        "total_class_sum": tot_class_sum,
        "per_image_counts": per_image_counts,
        "confidence_range_sums": confidence_range_sums,
    }


def check_for_exceptions(futs: List[Future]):
    for fut in futs:
        try:
            maybe_exc = fut.exception(timeout=1e-9)
        except TimeoutError:
            pass
        else:
            if maybe_exc is not None:
                raise maybe_exc


def plot_titration_curve(points, counts, plot_dir, model_name):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(
        f"{model_name} titration on {args.path_to_titration_yml.name}",
        fontsize=16,
    )

    ax[0].set_title("Total number of cells per titration point")
    ax[0].set_xlabel("Titration point")
    ax[0].set_xticks(points)
    ax[0].set_ylabel("Number of cells")
    ax[0].set_yscale("log")
    ax[0].plot(points, [c[:5].sum().item() for c in counts])

    ax[1].set_title("Normalized number of cells per class per titration point")
    ax[1].set_xlabel("Titration point")
    ax[1].set_xticks(points)
    ax[1].set_ylabel("parasitemia")
    ax[1].set_yscale("log")

    for i, class_name in enumerate(YOGO_CLASS_ORDERING[:5]):
        ax[1].plot(
            points,
            [c[i].item() / c[:5].sum().item() for c in counts],
            label=class_name,
        )
    ax[1].legend()

    plt.savefig(f"{(plot_dir / model_name).with_suffix('.png')}")


def plot_normalized_parasitemia(points, counts, plot_dir, model_name):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle(
        f"{model_name} titration on {args.path_to_titration_yml.name}",
        fontsize=16,
    )

    ax.set_title("Total number of parasitized cells per titration point")
    ax.set_xlabel("Titration point")
    ax.set_xticks(points)
    ax.set_ylabel(
        f"parasitemia (initial ground-truth parasitemia is {initial_parasitemia})"
    )
    ax.set_yscale("log")

    # index ring to gametocyte
    ax.plot(
        points,
        [c[1:5].sum().item() / c[:5].sum().item() for c in counts],
    )
    ax.plot(
        points, [initial_parasitemia / 2**i for i in range(len(titration_results))]
    )
    ax.legend(["YOGO predictions", "Ground Truth"])

    file_name = f"normalized_{Path(model_name).with_suffix('.png')}"
    plt.savefig(args.plot_dir / file_name)


def plot_normalized_parasitemia_multi_confidence_thresh(
    point, min_confidence_threshold_counts, plot_dir, model_name
):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle(
        f"{model_name} titration on {args.path_to_titration_yml.name}",
        fontsize=16,
    )

    ax.set_title(
        f"Normalized parasitemia at given minimum class confidence thresholds, titration point {point}"
    )
    ax.set_xlabel("Titration point")
    ax.set_xticks(points)
    ax.set_ylabel(
        f"parasitemia (initial ground-truth parasitemia is {initial_parasitemia})"
    )
    ax.set_yscale("log")

    dct = {
        round(min_confidence_threshold, 2): []
        for min_confidence_threshold in min_confidence_threshold_counts[0].keys()
    }

    for threshold_results in min_confidence_threshold_counts:
        for min_confidence_threshold, tot_counts in threshold_results.items():
            dct[round(min_confidence_threshold, 2)].append(tot_counts)

    for min_confidence_threshold, counts in dct.items():
        ax.plot(
            points,
            [
                (
                    c[1:5].sum().item() / c[:5].sum().item()
                    if c[:5].sum().item() > 0
                    else 0
                )
                for c in counts
            ],
            label=min_confidence_threshold,
        )

    ax.plot(
        points,
        [initial_parasitemia / 2**i for i in range(len(titration_results))],
        "--",
        label="Ground Truth",
    )

    ax.legend()

    file_name = f"normalized_by_threshold{Path(model_name).with_suffix('.png')}"
    plt.savefig(args.plot_dir / file_name)


def per_point_plot_normalized_per_image_counts(
    point, per_image_counts, plot_dir, model_name
):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle(
        f"{model_name} titration on {args.path_to_titration_yml.name}",
        fontsize=16,
    )

    ax.set_title(f"Normalized parasitemia per class for titration point {point}")
    ax.set_xlabel("image")
    ax.set_ylabel("parasitemia")
    ax.set_yscale("log")

    for i, class_name in enumerate(YOGO_CLASS_ORDERING[:5]):
        ax.plot(
            [c[i].item() / c[:5].sum().item() for c in per_image_counts],
            label=class_name,
        )
    ax.legend()

    file_name = f"per_img_parasitemia_{point}_{Path(model_name).with_suffix('.png')}"
    plt.savefig(args.plot_dir / file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("calculate a titration curve")
    parser.add_argument("path_to_pth", type=Path, help="path to yogo pth file")
    parser.add_argument(
        "path_to_titration_yml",
        type=Path,
        help="path to a titration dataset description file",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="directory for the plots (defaults to '.')",
        default=Path("."),
    )
    parser.add_argument(
        "--crop-height",
        type=float,
        help=(
            "crop images to the given fraction - e.g. `--crop-height 0.25` will crop "
            "images to a height of `round(height_org * 0.25)` (default to no cropping)"
        ),
    )
    parser.add_argument(
        "--min-confidence-threshold",
        type=float,
        help=(
            "minimum class confidence for the prediction to be counted - off by default"
        ),
    )
    args = parser.parse_args()

    if torch.multiprocessing.cpu_count() < 32 or not torch.cuda.is_available():
        warnings.warn(
            "Note: this will probably run out of memory and crash if you don't have enough cpus. Go to ondemand and request >32 CPUs and a GPU."
        )

    try:
        titration_points, initial_parasitemia, heatmap_masks = load_titration_yml(
            args.path_to_titration_yml
        )
    except KeyError as e:
        raise RuntimeError("invalid key in titration yml file") from e

    titration_results: Dict[
        str, Dict[str, Union[List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]]
    ] = {}

    futs: List[Future] = []
    tpe = ThreadPoolExecutor(max_workers=4)
    for (titration_point, path), (mask_titration_point, mask_path) in zip(
        titration_points.items(), heatmap_masks.items()
    ):
        if not titration_point == mask_titration_point:
            raise ValueError(
                f"Titration point for {path} did not match titration point for the corresponding mask: {mask_path}. Got {titration_point} and {mask_titration_point}. Double check the yaml file!"
            )
        if mask_path is not None:
            mask = np.load(mask_path)
        else:
            mask = None

        # let's error out asap
        check_for_exceptions(futs)
        # predict synchronously
        tn = predict(
            path_to_pth=args.path_to_pth,
            path_to_images=path,
            batch_size=128,
            use_tqdm=True,
            obj_thresh=0.5,
            iou_thresh=0.5,
            vertical_crop_height_px=(
                round(772 * args.crop_height) if args.crop_height is not None else None
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # and process results asynchronously
        fut = tpe.submit(
            process_prediction,
            tn,
            titration_point,
            titration_results,
            args.min_confidence_threshold,
            mask,
        )
        futs.append(fut)

    with utils.timing_context_manager(
        "waiting for yogo prediction processing to complete"
    ):
        tpe.shutdown(wait=True)
        check_for_exceptions(futs)

    datapoints = sorted(titration_results.items())
    points, results = zip(*datapoints)

    per_image_counts = [r["per_image_counts"] for r in results]
    counts = [r["total_class_sum"] for r in results]
    thresholded_counts = [r["confidence_range_sums"] for r in results]

    model_name = utils.guess_model_name(args.path_to_pth)

    args.plot_dir.mkdir(exist_ok=True, parents=True)

    plot_titration_curve(points, counts, args.plot_dir, model_name)
    plot_normalized_parasitemia(points, counts, args.plot_dir, model_name)
    plot_normalized_parasitemia_multi_confidence_thresh(
        points, thresholded_counts, args.plot_dir, model_name
    )

    N = int(math.log(len(points), 10) + 1)
    for point, per_image_count in zip(points, per_image_counts):
        per_point_plot_normalized_per_image_counts(
            f"{point:0{N}}", per_image_count, args.plot_dir, model_name
        )

    # Dump raw data
    try:
        raw_data_folder: Path = args.plot_dir / Path("raw_data")
        raw_data_folder.mkdir(exist_ok=True, parents=True)
        filename = str(raw_data_folder / f"{model_name}-thresholded_counts.csv")
        print(f"Writing raw data to csv: {filename}")
        with open(filename, "w") as csvfile:
            header = [
                "tpoint",
                "confidence_threshold",
                "actual_parasitemia",
                "perc_parasitemia",
                "healthy",
                "ring",
                "troph",
                "schizont",
                "gametocyte",
                "wbc",
                "misc",
            ]
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for i, vals in enumerate(thresholded_counts):
                for conf in vals:
                    # Counts for all classes for this confidence value (conf) for this titration point (i)
                    counts = np.asarray(vals[conf])

                    # numerator: rings + trophs + schizonts only
                    # denominator: healthy + rings + trophs + schizonts + gametocytes
                    perc_parasitemia = np.sum(counts[1:4]) / np.sum(counts[:5])
                    actual_parasitemia = initial_parasitemia * (2 ** (i - 1))

                    row = np.concatenate(
                        ([i + 1, conf, actual_parasitemia, perc_parasitemia], counts)
                    )

                    writer.writerow(list(row))

    except Exception as e:
        print(f"Error when dumping raw data: {e}!")
