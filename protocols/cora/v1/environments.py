from abc import abstractmethod
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
from simulator.environment import BaseSimulationEnvironment
from simulator.environment.brownian_environment import BrownianSimulationEnvironment
from simulator.models.event_info import EventInfo
from simulator.utilities.price_data import PriceData, PriceDataItem


class BaseCoraEnvironment(BaseSimulationEnvironment):
    @abstractmethod
    def get_price(self) -> float:
        pass

    @abstractmethod
    def get_price_data(self) -> List[PriceDataItem]:
        pass

    @abstractmethod
    def get_price_history(self, delta: timedelta) -> List[PriceDataItem]:
        pass

    @abstractmethod
    def get_price_for_timestamps(self, timestamps: List[int]) -> List[float]:
        pass

    def _take_step(self, time_step: timedelta) -> List[EventInfo]:
        current_price = self.get_price()
        current_time = self.get_time()
        message = (
            f"{current_time}: Taking step of {time_step.seconds} seconds, "
            f"current price is {current_price:.4f}"
        )
        return [
            EventInfo(
                message=message,
                time=current_time,
                type="environment_step",
                extra={
                    "time_step": time_step.seconds,
                    "current_price": current_price,
                },
            )
        ]


class HistoricalCoraEnvironment(BaseCoraEnvironment):
    def __init__(self, symbol: str):
        self._symbol = symbol
        self._price_data = []

    def _load_data_until(self, end: datetime) -> None:
        price_data = PriceData(self._symbol).get_data(int(end.timestamp()))
        self._price_data = price_data

        self._timestamps = np.array([item.time for item in price_data])
        self._prices = np.array([item.price for item in price_data])
        self._previous_interp = lambda t: np.maximum(
            np.searchsorted(self._timestamps, t, side="right") - 1, 0
        )

    def get_price_history(self, delta: timedelta) -> List[PriceDataItem]:
        idx_start = self._previous_interp(int((self._time - delta).timestamp()))
        idx_end = self._previous_interp(int(self._time.timestamp()))
        return self._price_data[idx_start : idx_end + 1]

    def get_price_for_timestamps(self, timestamps: List[int]) -> List[float]:
        idxs = self._previous_interp(timestamps)
        return self._prices[idxs]

    def get_price_data(self) -> List[PriceDataItem]:
        return self._price_data

    def get_price(self) -> float:
        return self.get_searchsorted_price_for_timestamp(int(self._time.timestamp()))

    @lru_cache
    def get_searchsorted_price_for_timestamp(self, timestamp: int) -> float:
        return self._prices[self._previous_interp(timestamp)]


class BrownianCoraEnvironment(HistoricalCoraEnvironment, BrownianSimulationEnvironment):
    def __init__(
        self, symbol: str, volatility_factor: float = 1.0, zero_mu: bool = True
    ):
        super().__init__(symbol)
        self._volatility_factor = volatility_factor
        self._zero_mu = zero_mu

    def _load_data_until(self, end: datetime) -> None:
        super()._load_data_until(self._time)

        price_data = self._generate_brownian_continuation_until(end)

        self._price_data = price_data

        self._timestamps = np.array([item.time for item in price_data])
        self._prices = np.array([item.price for item in price_data])
        self._previous_interp = lambda t: np.maximum(
            np.searchsorted(self._timestamps, t, side="right") - 1, 0
        )

    def _generate_brownian_continuation_until(
        self, end: datetime
    ) -> List[PriceDataItem]:
        needed_hours = pd.date_range(start=self._time, end=end, freq="H")

        brownian_series = self._generate_brownian_continuation_from(
            [[item.price for item in self._price_data]],
            len(needed_hours),
            zero_mu=self._zero_mu,
            sigma_factor=self._volatility_factor,
            rng=self._rng,
        )

        return self._price_data + [
            PriceDataItem(time=int(hour.timestamp()), price=price)
            for (hour, price) in zip(needed_hours, brownian_series.flatten())
        ]


class ShuffleCoraEnvironment(BaseCoraEnvironment):
    pass
