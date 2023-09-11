from itertools import groupby
from typing import Iterable, Callable, Optional, TypeVar, Dict
import numpy as np

Element = TypeVar("Element")


def safe_divide(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


class MetricBinner:
    def __init__(
        self,
        name: str,
        distfunc: Callable[[Element], float],
        bins: Iterable[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ):
        self.name = name
        self.distfunc = distfunc
        self.bins = np.sort(np.array(bins))

    def _get_bin_indexes(self, values: Iterable[float]) -> np.ndarray:
        values = np.array(values)
        values = values[np.logical_and(values >= self.bins[0], values <= self.bins[-1])]
        return np.searchsorted(self.bins[:-1], values, "right") - 1

    def count(self, elements: Iterable[Element]) -> Dict[str, int]:
        distribution = np.array(list(map(self.distfunc, elements)))
        histogram = np.histogram(distribution, bins=self.bins)[0]
        return {
            f"{self.name}_{bin_start}_{bin_end}": int(count)
            for bin_start, bin_end, count in zip(
                self.bins[:-1], self.bins[1:], histogram
            )
        }

    def aggregate(
        self,
        elements: Iterable[Element],
        aggfunc: Callable[[Iterable[Element]], float] = len,
    ) -> Dict[str, float]:
        distribution = list(map(self.distfunc, elements))
        bin_indexes = self._get_bin_indexes(distribution)
        grouping = {i: [] for i in range(len(self.bins) - 1)}
        for element, idx in zip(elements, bin_indexes):
            grouping[idx].append(element)
        aggs = [aggfunc(grouping.get(idx, [])) for idx in range(len(self.bins) - 1)]

        return {
            f"{self.name}_{bin_start}_{bin_end}": agg
            for bin_start, bin_end, agg in zip(self.bins[:-1], self.bins[1:], aggs)
        }


def apply_metric_binners(
    elements: Iterable[Element],
    binners: Iterable[MetricBinner],
    prefix: str,
    aggregator: Optional[Callable[[Iterable[Element]], float]] = None,
):
    if (aggregator is None) or (aggregator is len):
        return {
            f"{prefix}-{k}": v for b in binners for k, v in b.count(elements).items()
        }

    return {
        f"{prefix}-{k}": v
        for b in binners
        for k, v in b.aggregate(elements, aggregator).items()
    }
