#! /usr/bin/env python3


import cmd
import IPython
import argparse

import torch

import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Callable, Any, Optional

from yogo.utils import format_preds

from lfm_data_utilities import utils
from lfm_data_utilities.malaria_labelling.labelling_constants import CLASSES


"""
This is a tool specifically built for map-reducing over a large number
of large tensors.


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
        list(prediction_tensor_paths)[:2],
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


ImgReductionType = Callable[
    [
        torch.Tensor,
    ],
    torch.Tensor,
]


class ImgReduction:
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
        return nonzero_mean(ImgReduction.predicted_confidence(prediction)).unsqueeze(
            dim=0
        )

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
        return (
            ImgReduction.predicted_confidence(prediction)
            .ceil()
            .sum(dim=0)
            .unsqueeze(dim=0)
        )


RunReductionType = Callable[
    [
        List[torch.Tensor],
    ],
    torch.Tensor,
]


class RunReduction:
    @staticmethod
    def cat(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values)

    @staticmethod
    def nonzero_mean(values: List[torch.Tensor]) -> torch.Tensor:
        return nonzero_mean(torch.cat(values))

    @staticmethod
    def mean(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values).mean(dim=0)

    @staticmethod
    def median(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values).median(dim=0).values

    @staticmethod
    def min(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values).min(dim=0).values

    @staticmethod
    def max(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values).max(dim=0).values

    @staticmethod
    def sum(values: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values).sum(dim=0)


RunSetReductionType = Callable[
    [
        Dict[Path, torch.Tensor],
    ],
    Any,
]


class RunSetReduction:
    @staticmethod
    def id(values: Dict[Path, torch.Tensor]) -> Dict[Path, torch.Tensor]:
        return values

    @staticmethod
    def min_n(
        values: Dict[Path, torch.Tensor], n: int, class_idx: int = 0
    ) -> Dict[Path, torch.Tensor]:
        return dict(sorted(values.items(), key=lambda v: v[1][class_idx].item())[:n])

    @staticmethod
    def max_n(
        values: Dict[Path, torch.Tensor], n: int, class_idx: int = 0
    ) -> Dict[Path, torch.Tensor]:
        return dict(
            sorted(values.items(), key=lambda v: v[1][class_idx].item(), reverse=True)[
                :n
            ]
        )


def execute_arrr(
    path_tensor_map: Dict[Path, List[torch.Tensor]],
    img_reduction: ImgReductionType,
    run_reduction: RunReductionType,
    run_set_reduction: RunSetReductionType,
) -> Any:
    """execute array reduce reduce reduce. lots of parallelization opportunity here."""
    path_modified_tensor_map = {
        path: run_reduction([img_reduction(img_tensor) for img_tensor in run_tensor])
        for path, run_tensor in tqdm(path_tensor_map.items())
    }
    return run_set_reduction(path_modified_tensor_map)


def get_user_defined_methods(class_):
    return [method for method in class_.__dict__ if callable(getattr(class_, method))]


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

        self.prev_results: Optional[Dict[str, torch.Tensor]] = dict()
        self.prev_op_name: Optional[str] = None

        self._df = pd.DataFrame(index=self.path_tensor_map.keys())
        self._df.index.name = "path"

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

    def do_append(self, arg):
        "append previous results to the internal dataframe (to be saved later with the `save` command)"
        if self.prev_results is None:
            print("no previous results to append")
            return

        mi = pd.MultiIndex.from_product([[self.prev_op_name], CLASSES])
        df_new = pd.DataFrame(columns=mi, index=self.path_tensor_map.keys())

        # probably can set this without a loop, but I have no clue how to do it
        for path, tensor in self.prev_results.items():
            for class_idx, class_name in enumerate(CLASSES):
                df_new.loc[path, (self.prev_op_name, class_name)] = tensor[
                    class_idx
                ].item()

        self._df = pd.concat([self._df, df_new], axis=1, join="inner")

    def do_save(self, arg):
        """
        save results to csv

        usage: save <filename>
        """
        if arg == "":
            print("usage: save <filename>")
            return

        self._df.to_csv(arg)

    def do_show_df(self, _):
        "show the first several rows of the dataframe"
        print(self._df.head())

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

    def do_mean_class_probability(self, arg):
        "print the mean per-class confidence for each dataset"
        self.prev_results = execute_arrr(
            path_tensor_map=self.path_tensor_map,
            img_reduction=ImgReduction.mean_predicted_confidence,
            run_reduction=RunReduction.nonzero_mean,
            run_set_reduction=RunSetReduction.id,
        )
        self.prev_op_name = "mean_class_probability"
        self.pretty_print_dict(self.prev_results)

    def do_count_class(self, arg):
        """
        print the number of cells per class for each dataset
        """
        self.prev_results = execute_arrr(
            path_tensor_map=self.path_tensor_map,
            img_reduction=ImgReduction.count_class,
            run_reduction=RunReduction.sum,
            run_set_reduction=RunSetReduction.id,
        )
        self.prev_op_name = "count_class"
        self.pretty_print_dict(self.prev_results)

    def do_query(self, arg):
        maybe_methods = self._parse_args(arg)
        if maybe_methods is None:
            return

        (
            img_reduction_method,
            run_reduction_method,
            run_set_reduction_method,
        ) = maybe_methods

        try:
            self.prev_results = execute_arrr(
                path_tensor_map=self.path_tensor_map,
                img_reduction=img_reduction_method,
                run_reduction=run_reduction_method,
                run_set_reduction=run_set_reduction_method,
            )
        except Exception as e:
            print(f"error executing arrr: {e}")
            return
        self.pretty_print_dict(self.prev_results)

    do_query.__doc__ = f"""
        general query

        format is:
            <img reduction> <run reduction> <run set reduction>

        options for img_reduction are
            {get_user_defined_methods(ImgReduction)}

        options for run_reduction are
            {get_user_defined_methods(RunReduction)}

        options for run_set_reduction are
            {get_user_defined_methods(RunSetReduction)}

        some options for each reduction won't mix - let axel know if there
        are any specific operations you want to do that aren't supported
    """

    def do_quit(self, arg):
        "quit the program"
        return True

    def do_embed(self, _):
        """
        embed into an ipython shell - note, you can break things here, but it will
        be useful for e.g. plotting results from the dataframe
        """

        IPython.embed()

    def _parse_args(
        self, arg
    ) -> Optional[Tuple[ImgReductionType, RunReductionType, RunSetReductionType]]:
        """
        format of args is expected to be

            <img reduction> <run reduction> <run set reduction>
        """
        arguments = [arg.strip() for arg in arg.split(" ") if arg.strip()]
        if len(arguments) != 3:
            print("invalid number of arguments")
            return None

        img_reduction_method = getattr(ImgReduction, arguments[0], None)
        if img_reduction_method is None:
            print(f"invalid img reduction method {arguments[0]}")
            return None

        run_reduction_method = getattr(RunReduction, arguments[1], None)
        if run_reduction_method is None:
            print(f"invalid run reduction method {arguments[1]}")
            return None

        run_set_reduction_method = getattr(RunSetReduction, arguments[2], None)
        if run_set_reduction_method is None:
            print(f"invalid run set reduction method {arguments[2]}")
            return None

        return img_reduction_method, run_reduction_method, run_set_reduction_method

    def parse_to_float(self, arg: Any) -> Optional[float]:
        try:
            return float(arg)
        except ValueError:
            return None

    def emptyline(self):
        "on empty line, do nothing"

    def pretty_print_dict(self, d: Dict[Any, Any]):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.tolist()
                if not isinstance(v[0], int):
                    v = [round(x, 2) for x in v]

            print(f"{k}: {v}")


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
