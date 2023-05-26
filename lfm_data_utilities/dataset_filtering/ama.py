#! /usr/bin/env python3


import os
import cmd
import argparse

import torch

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor

from yogo.utils import format_preds
from lfm_data_utilities import utils


"""
This is a tool specifically built for map-reducing over a large number
of large tensors.

Types of questions I can answer!

(notation: map is "->", reduce is "|->")

- find min 100 images ranked by mean class confidence
    - map: tensor |-> predictions -> class confidences |-> mean
    - reduce: mean |-> min
- rank all runs by mean class confidence
    - map: tensor |-> predictions -> class confidences |-> mean
    - reduce: mean |-> rank
"""


def load_predictions_into_memory(
    prediction_tensor_paths: Sequence[Path],
    objectness_threshold: float,
    iou_threshold: float = 0.5,
) -> Dict[Path, List[torch.Tensor]]:
    """This is the main expensive operation. Loading all tensors sequentially takes roughly 15 minutes (w/ single-threaded processing,
    most likely much faster w/ multithreading).

    The total size of tensors (as of 05/26/2023) is 660 Gb. Most of this is just 0s (due to the format of prediction tensors),
    so we 'format_preds' to reduce the size of tensors. However, whenever we want to change the objectness threshold, we
    have to rerun this function.
    """

    def load_tensor(path: Path) -> Tuple[Path, List[torch.Tensor]]:
        full_tensor = torch.load(path)
        return (
            path,
            [
                format_preds(
                    pred, thresh=objectness_threshold, iou_thresh=iou_threshold
                )
                for pred in full_tensor
            ],
        )

    path_prediction_pairs = utils.multithread_map_unordered(
        prediction_tensor_paths,
        fn=load_tensor,
        verbose=True,
        realize=True,
    )

    return dict(path_prediction_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ask Me Anything*")
    parser.add_argument(
        "dense_data_dir",
        help="path to 'dense data' dir - ie dir that has directories  with 'yogo_predictions.pt'",
        type=Path,
    )
    parser.add_argument(
        "--objectness",
        help="objectness threshold (default: 0.5)",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    prediction_tensor_paths = args.dense_data_dir.glob("*/yogo_predictions.pt")

    with utils.timing_context_manager("Loading tensors"):
        tensors = load_predictions_into_memory(
            prediction_tensor_paths, objectness_threshold=args.objectness
        )

    tot_size = sum([ar.storage().nbytes() for t in tensors.values() for ar in t])
    print(tot_size / 1e9, "Gb")
