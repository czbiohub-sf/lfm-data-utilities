#! /usr/bin/env python3


import cmd
import argparse
import operator

import torch

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Callable, Any, Optional

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


In this file for shape documentation, pred-size is the size of the prediction dimension.
That is, pred-size == 12, since len(xc, yc, w, h, t0, prob_healthy, prob_ring, ...) = 12 (since we have 7 classes)
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
        list(prediction_tensor_paths)[:5],
        fn=load_tensor,
        verbose=True,
        realize=True,
    )

    return dict(path_prediction_pairs)


def nonzero_mean(ten: torch.Tensor, dim=0, nan=0.0) -> torch.Tensor:
    """mean of non-zero elements along dimension dim."""
    sum_ = ten.sum(dim=dim)
    nonzero = (ten != 0).sum(dim=dim)
    return torch.nan_to_num(sum_ / nonzero, nan=nan)


class PerImgReduction:
    @staticmethod
    def predicted_confidence(prediction):
        """
        given a (N, pred-size) tensor of YOGO predictions, return a tensor of shape
        (N, num-classes) with all values that aren't the max for each row set to 0.

        If the class-component* of the prediction tensor is

        [[0.6, 0.2],
         [0.2, 0.8],
         [0.1, 0.9]]

         this returns

        [[0.6, 0.0],
         [0.0, 0.8],
         [0.0, 0.9]]

         * the class-component is pred[:, 5:], if pred.shape == (N, pred-size)
        """
        assert (
            prediction.ndim == 2
        ), f"prediction tensor must be 2-dimensional, got {prediction.ndim}"
        num_classes = prediction.shape[1] - 5
        class_probabilities = prediction[:, 5:]
        class_predictions = class_probabilities.argmax(dim=1)
        return class_probabilities * torch.nn.functional.one_hot(
            class_predictions, num_classes=num_classes
        )

    @staticmethod
    def mean_predicted_confidence(prediction):
        """
        given a (N, pred-size) tensor of YOGO predictions, return a tensor of shape
        (1, num_classes) with each value being the mean of predicted classes. So, if
        the class-component of the prediction tensor is

        [[0.6, 0.2],
         [0.2, 0.8],
         [0.1, 0.9]]

        this returns

        [0.6, 0.85]

        since the mean of the first column (ignoring values that are not the best prediction)
        is 0.6, and the mean of the second column is 0.85.
        """
        return nonzero_mean(PerImgReduction.predicted_confidence(prediction))

    @staticmethod
    def count_class(prediction):
        """
        given a (N, pred-size) tensor of YOGO predictions, return a tensor of shape
        (1, num_classes) with each value being the number of times that class was predicted.
        Easy, eh!

        If the class-component of the prediction tensor is

        [[0.6, 0.2],
         [0.2, 0.8],
         [0.1, 0.9]]

        this returns

        [1, 2]

        """
        return PerImgReduction.predicted_confidence(prediction).ceil().sum(dim=0)


class PerRunReduction:
    @staticmethod
    def stack(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values)

    @staticmethod
    def mean(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean(dim=0)

    @staticmethod
    def median(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).median(dim=0)

    @staticmethod
    def min(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).min(dim=0)

    @staticmethod
    def max(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).max(dim=0)

    @staticmethod
    def sum(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).sum(dim=0)


class RunSetReduction:
    @staticmethod
    def id(values: Dict[Path, torch.Tensor]) -> Dict[Path, torch.Tensor]:
        return values

    @staticmethod
    def min_n(values: Dict[Path, torch.Tensor], n: int) -> Dict[Path, torch.Tensor]:
        return dict(sorted(values.items(), key=operator.itemgetter(1))[:n])

    @staticmethod
    def max_n(values: Dict[Path, torch.Tensor], n: int) -> Dict[Path, torch.Tensor]:
        return dict(
            sorted(values.items(), key=operator.itemgetter(1), reverse=True)[:n]
        )


# geez! the autoformatting of callables is brutal
def execute_arrr(
    path_tensor_map: Dict[Path, List[torch.Tensor]],
    per_img_reduction: Callable[
        [
            torch.Tensor,
        ],
        torch.Tensor,
    ],
    per_run_reduction: Callable[
        [
            List[torch.Tensor],
        ],
        torch.Tensor,
    ],
    per_run_set_reduction: Callable[
        [
            Dict[Path, torch.Tensor],
        ],
        Any,
    ],
) -> Any:
    """execute array reduce reduce reduce. lots of parallelization opportunity here."""
    path_modified_tensor_map = {
        path: per_run_reduction(
            [per_img_reduction(img_tensor) for img_tensor in run_tensor]
        )
        for path, run_tensor in path_tensor_map.items()
    }
    return per_run_set_reduction(path_modified_tensor_map)


class ARRRShell(cmd.Cmd):
    intro = "type ? or help for a list of commands"
    prompt = "ARRR | "

    def __init__(
        self,
        path_tensor_map: Dict[Path, List[torch.Tensor]],
        objectness_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.path_tensor_map = path_tensor_map
        self.objectness_threshold = objectness_threshold
        self.iou_threshold = iou_threshold

    def parse_to_float(self, arg: Any) -> Optional[float]:
        try:
            return float(arg)
        except ValueError:
            return None

    def emptyline(self):
        "on empty line, do nothing"
        pass

    def pretty_print_dict(self, d: Dict[Any, Any]):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.tolist()
                if not isinstance(v[0], int):
                    v = [round(x, 2) for x in v]

            print(f"{k}: {v}")

    def do_why(self, _):
        "why is this called ARRR?"
        print(
            "\nArray Reduce Reduce Reduce (ARRR)\n"
            "---------------------------------\n\n"
            "Three levels of reductions:\n"
            "    1. per-image reduction over cell predictions\n"
            "    2. per-run reduction over per-image reductions\n"
            "    3. per-run-set reduction over per-run reductions\n"
        )

    def do_set_objectness(self, arg):
        """
        set objectness threshold for predictions - i.e. '0.5' keeps all predictions w/ predicted objectness >= 0.5
        """
        v = self.parse_to_float(arg)
        if v is None:
            print(f"objectness must be a float; got {arg}")
            return
        elif not 0 <= v <= 1:
            print(f"objectness must be between 0 and 1; got {arg}")
            return

        self.objectness_threshold = v
        self.path_tensor_map = load_predictions_into_memory(
            self.path_tensor_map.keys(),
            objectness_threshold=self.objectness_threshold,
            iou_threshold=self.iou_threshold,
        )

    def do_set_IoU(self, arg):
        """
        set IoU threshold for non-maximal supression (i.e. doubled bounding boxes) - i.e. '0.5' will discard doubled
        bounding boxes w/ IoU greater than 0.5 until only one remains
        """
        v = self.parse_to_float(arg)
        if v is None:
            print(f"IoU Threshold must be a float; got {arg}")
            return
        elif not 0 <= v <= 1:
            print(f"IoU Threshold must be between 0 and 1; got {arg}")
            return

        self.iou_threshold = v
        self.path_tensor_map = load_predictions_into_memory(
            self.path_tensor_map.keys(),
            objectness_threshold=self.objectness_threshold,
            iou_threshold=self.iou_threshold,
        )

    def do_per_dataset_mean_class_probability(self, arg):
        "print the mean per-class confidence for each dataset"
        self.pretty_print_dict(
            execute_arrr(
                path_tensor_map=self.path_tensor_map,
                per_img_reduction=PerImgReduction.mean_predicted_confidence,
                per_run_reduction=PerRunReduction.mean,
                per_run_set_reduction=RunSetReduction.id,
            )
        )

    def do_per_dataset_class_count(self, arg):
        self.pretty_print_dict(
            execute_arrr(
                path_tensor_map=self.path_tensor_map,
                per_img_reduction=PerImgReduction.count_class,
                per_run_reduction=PerRunReduction.sum,
                per_run_set_reduction=RunSetReduction.id,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ARRR")
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
