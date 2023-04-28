#! /usr/bin/env python3

from typing import Dict
from abc import ABC, abstractmethod


CSVRow = Dict[str, str]


class Evaluator(ABC):
    """each evaluator takes a row and accumulates it over time"""

    @abstractmethod
    def accumulate(self, row: CSVRow):
        ...

    @abstractmethod
    def compute(self) -> float:
        ...

    @abstractmethod
    def metric_passed(self) -> bool:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class RangeBooleanEvaluator(Evaluator):
    def __init__(self, failrate: float, center: float, step: float):
        assert 0 <= failrate <= 1, f"failrate must be in [0,1] (got {failrate})"
        self.failrate = failrate
        self.step = [center - step, center + step]
        self.sum: float = 0.0
        self.tot_num_samples: int = 0

    def compute(self) -> float:
        return self.sum / self.tot_num_samples

    def metric_passed(self) -> bool:
        return self.sum / self.tot_num_samples >= self.failrate

    def reset(self) -> None:
        self.sum = 0
        self.tot_num_samples = 0


class SSAFBooleanEvaluator(RangeBooleanEvaluator):
    def __init__(self, failrate: float, step: float):
        super().__init__(failrate, 0.0, step)

    def __repr__(self):
        return f"SSAFBooleanEvaluator(accumulated rate {self.compute():.4f})"

    def accumulate(self, row: CSVRow):
        value = float(row["autofocus"])
        self.sum += int(self.step[0] <= value <= self.step[1])
        self.tot_num_samples += 1
