#! /usr/bin/env python3

import csv
import argparse

from pathlib import Path
from typing import Sequence

from lfm_data_utilities import utils
from lfm_data_utilities.dataset_filtering import evaluators as ev


""" Evaluation of dense metrics

What qualifies as a good run?

improvements:
    - add more evaluators
    - multithread options
"""


def eval_data_csv(
    data_path: Path, evaluator: ev.Evaluator
) -> ev.Evaluator:
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evaluator.accumulate_row(row)

    return evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate dense data from runs")
    parser.add_argument(
        "path_to_dense_data_dir", type=Path, help="path to directory of 'dense data'"
    )
    parser.add_argument(
        "--ssaf-failrate",
        type=float,
        default=0.9,
        help="percent of results (in [0,1]) that must succeed for a pass",
    )
    parser.add_argument(
        "--flowrate-failrate",
        type=float,
        default=0.9,
        help="percent of results (in [0,1]) that must succeed for a pass (default 0.9)",
    )
    parser.add_argument(
        "--ssaf-step",
        type=float,
        default=2.0,
        help="step size for a pass - i.e. a value must be in [-val, val] (default 2.0)",
    )

    args = parser.parse_args()

    if (
        not args.path_to_dense_data_dir.exists()
        or not args.path_to_dense_data_dir.is_dir()
    ):
        raise ValueError(
            f"{args.path_to_dense_data_dir} does not exist or is not a dir"
        )

    evaluators = ev.EvaluatorCollection(
        ev.SSAFEvaluator(args.ssaf_failrate, args.ssaf_step),
        ev.FlowrateEvaluator(args.ssaf_failrate, args.ssaf_step),
        ev.YOGOEvaluator(args.ssaf_failrate, args.ssaf_step),
    )

    with utils.timing_context_manager("evals"):
        for data_file in args.path_to_dense_data_dir.rglob("data.csv"):
            eval_data_csv(data_file, evaluator=evaluators)

    for evalu in evaluators:
        print(
            f"{evalu} success rate = {evalu.compute():.3f} pass? {evalu.metric_passed()}"
        )
