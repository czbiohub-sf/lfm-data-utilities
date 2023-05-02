#! /usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Sequence

from lfm_data_utilities.malaria_labelling.labelling_constants import CLASSES


CSVRow = Dict[str, str]


class Evaluator(ABC):
    """each evaluator takes a row and accumulates it over time"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.compute()})"

    @abstractmethod
    def accumulate(self, value: Any) -> None:
        """
        accumulate a specific value - main method for accumulating
        the relevant data for the subclass
        """
        ...

    @abstractmethod
    def accumulate_row(self, row: CSVRow) -> None:
        """
        accumulate a row - in general, this will pick the relevant
        data from the row and pass it to accumulate
        """
        ...

    @abstractmethod
    def compute(self) -> Any:
        """compute the metric"""
        ...

    @abstractmethod
    def metric_passed(self) -> bool:
        """did the metric pass?"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """take a guess! resets the evaluator to it's initial state"""
        ...


class EvaluatorCollection(Evaluator):
    """represents a collection of evaluators"""
    def __init__(self, *args: Evaluator) -> None:
        self.evaluators = args

    def accumulate(self, value: Any) -> None:
        raise NotImplementedError(
            "EvaluatorCollection does not support accumulate - use accumulate_row instead"
        )

    def accumulate_row(self, row: CSVRow) -> None:
        for evaluator in self.evaluators:
            evaluator.accumulate_row(row)

    def compute(self) -> List[Any]:
        return [evaluator.compute() for evaluator in self.evaluators]

    def per_metrics_pass_fail(self) -> List[bool]:
        return [evaluator.metric_passed() for evaluator in self.evaluators]

    def metric_passed(self) -> bool:
        return all(self.per_metrics_pass_fail())

    def reset(self) -> None:
        for evaluator in self.evaluators:
            evaluator.reset()


class RangeEvaluator(Evaluator):
    """Generic evaluator used to check that a value is in an absolute range

    Concretely, if the total fraction of samples that are in the range
    [center - step, center + step] is greater than the failrate, then the
    metric passes.
    """

    def __init__(self, failrate: float, center: float, step: float) -> None:
        assert 0 <= failrate <= 1, f"failrate must be in [0,1] (got {failrate})"
        self.failrate = failrate
        self.step = (center - step, center + step)
        self.sum: float = 0.0
        self.tot_num_samples: int = 0

    def accumulate(self, value: float) -> None:
        self.sum += int(self.step[0] <= value <= self.step[1])
        self.tot_num_samples += 1

    def compute(self) -> float:
        return self.sum / self.tot_num_samples

    def metric_passed(self) -> bool:
        return self.sum / self.tot_num_samples >= self.failrate

    def reset(self) -> None:
        self.sum = 0
        self.tot_num_samples = 0


class FractionRangeEvaluator(RangeEvaluator):
    """Generic evaluator used to check that a value is in a relative range

    Concretely, if the total fraction of samples that are in the range
    [center - fraction * center, center + fraction * center] is greater than
    the failrate, then the metric passes.
    """

    def __init__(self, failrate: float, center: float, fraction: float) -> None:
        assert (
            0 <= fraction <= 1
        ), f"fractional percent must be in [0,1] (got {fraction})"
        super().__init__(failrate, center, 0)
        # overwrite the step range
        self.step = (center - fraction * center, center + fraction * center)


class SSAFEvaluator(RangeEvaluator):
    def __init__(self, failrate: float, step: float) -> None:
        super().__init__(failrate, 0.0, step)

    def __repr__(self) -> str:
        return f"SSAFEvaluator(accumulated rate {self.compute():.4f})"

    def accumulate_row(self, row: CSVRow) -> None:
        value = float(row["autofocus"])
        self.accumulate(value)


class FlowrateEvaluator(FractionRangeEvaluator):
    def __init__(
        self,
        failrate: float,
        flowrate: float,
        relative_range: float,
        flowrate_confidence_threshold: float,
    ) -> None:
        super().__init__(failrate, flowrate, relative_range)
        self.flowrate_confidence_threshold = flowrate_confidence_threshold

    def accumulate_row(self, row: CSVRow) -> None:
        flowrate_dx = float(row["flowrate_dx"])
        flowrate_dy = float(row["flowrate_dy"])
        flowrate_confidence = float(row["flowrate_confidence"])
        if flowrate_confidence > self.flowrate_confidence_threshold:
            value = (flowrate_dx**2 + flowrate_dy**2) ** 0.5
            self.accumulate(value)
        else:
            self.tot_num_samples += 1


class YOGOEvaluator(Evaluator):
    def __init__(self) -> None:
        self.classes: List[str] = CLASSES
        self.class_counts: Dict[str,float]  = {c: 0.0 for c in self.classes}

    def accumulate(self, kv: Tuple[str, float]) -> None:
        key, value = kv
        self.class_counts[key] += value

    def accumulate_row(self, row: CSVRow) -> None:
        for c in self.classes:
            self.accumulate((c, float(row[c])))

    def compute(self) -> Any:
        return self.class_counts

    def metric_passed(self) -> bool:
        return True

    def reset(self) -> None:
        self.class_counts = {c: 0.0 for c in self.classes}
