#! /usr/bin/env python3

import csv
import argparse

from pathlib import Path
from typing import List, Dict
from abc import ABC, abstractmethod


""" Evaluation of dense metrics

What qualifies as a good run?
"""


class Evaluator(ABC):
    @abstractmethod
    def accumulate(self, value: float):
        ...

    @abstractmethod
    def compute(self) -> float:
        ...

    @abstractmethod
    def metric_passed(self) -> bool:
        ...


class RangeBooleanEvaluator(Evaluator):
    def __init__(self, failrate: float, center: float, step: float):
        assert 0 <= failrate <= 1, f"failrate must be in [0,1] (got {failrate})"
        self.failrate = failrate
        self.step = [center - step, center + step]
        self.sum = 0

    def accumulate(self, value: float):
        self.sum += int(self.step[0] <= value <= self.step[1])

    def compute(self) -> float:
        return self.sum

    def metric_passed(self) -> bool:
        return self.sum >= self.failrate


class SSAFBooleanEvaluator(RangeBooleanEvaluator):
    def __init__(self, failrate: float, step: float):
        super().__init__(failrate, 0.0, step)



def eval_data_csv(data_path: Path, evaluators: Dict[str, Evaluator]) -> Dict[str, Evaluator]:
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, evaluator in evaluators.items():
                evaluator.accumulate(float(row[key]))

    return evaluators


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate dense data from runs")
    parser.add_argument("path_to_dense_data_dir", type=Path, help="path to directory of 'dense data'")
    parser.add_argument("--ssaf-failrate", type=float, default=0.9, help="percent of results (in [0,1]) that must succeed for a pass")
    parser.add_argument("--flowrate-failrate", type=float, default=0.9, help="percent of results (in [0,1]) that must succeed for a pass (default 0.9)")
    parser.add_argument("--ssaf-step", type=float, default=2.0, help="step size for a pass - i.e. a value must be in [-val, val] (default 2.0)")

    args = parser.parse_args()

    if not args.path_to_dense_data_dir.exists() or not args.path_to_dense_data_dir.is_dir():
        raise ValueError(f"{args.path_to_dense_data_dir} does not exist or is not a dir")

    for data_csv in args.path_to_dense_data_dir.rglob("data.csv"):
        pass
