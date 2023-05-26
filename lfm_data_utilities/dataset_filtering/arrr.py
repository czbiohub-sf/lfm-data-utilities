#! /usr/bin/env python3


import os
import cmd
import argparse

import torch

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple, Callable

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


Array Reduce Reduce Reduce
--------------------------

Three levels of reductions:
    1. per-image reduction over cell predictions
    2. per-run reduction over per-image reductions
    3. per-run-set reduction over per-run reductions


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
        list(prediction_tensor_paths)[:1],
        fn=load_tensor,
        verbose=True,
        realize=True,
    )

    return dict(path_prediction_pairs)



def mean_class_confidence(
    predictions: List[torch.Tensor],
) -> torch.Tensor:
    """ for a run, return mean confidence for each class, given the class was the argmax
    """
    confidences = []
    for prediction in predictions:
        num_classes = prediction.shape[1] - 5
        class_probabilities = prediction[:, 5:]
        class_predictions = class_probabilities.argmax(dim=1)
        confidences.append(
            (class_probabilities *
            torch.nn.functional.one_hot(
                class_predictions,
                num_classes=num_classes
            )).mean(dim=0)
        )
    return torch.stack(confidences)


def per_dataset_quantity(
    path_to_tensors: Dict[Path, List[torch.Tensor]],
    fn: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Dict[Path, torch.Tensor]:
    return {
        path: fn(tensors)
        for path, tensors in path_to_tensors.items()
    }



class ARRRShell(cmd.Cmd):
    intro = "type ? or help for a list of commands"
    prompt = "ARRR> "

    def __init__(self, path_to_tensors: Dict[Path, List[torch.Tensor]], objectness_threshold: float=0.5, iou_threshold: float=0.5):
        super().__init__()
        self.path_to_tensors = path_to_tensors
        self.objectness_threshold = objectness_threshold
        self.iou_threshold = iou_threshold

    def parse_to_float(self, arg):
        try:
            return float(arg)
        except ValueError:
            raise ValueError(f"could not parse '{arg}' to float")

    def emptyline(self): pass

    def do_set_objectness(self, arg):
        arg = self.parse_to_float(arg)
        if not 0 <= arg <= 1:
            raise ValueError("objectness must be between 0 and 1")
        self.path_to_tensors = load_predictions_into_memory(
            self.path_to_tensors.keys(),
            objectness_threshold=arg,
            iou_threshold=self.iou_threshold,
        )

    def do_per_dataset_mean_confidence(self, arg):
        confidences = per_dataset_quantity(
            self.path_to_tensors,
            mean_class_confidence,
        )
        for path, confidence in confidences.items():
            print(path, confidence.mean(dim=0))


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

    shell = ARRRShell(tensors, objectness_threshold=args.objectness)
    shell.cmdloop()
