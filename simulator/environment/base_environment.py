from datetime import datetime, timedelta
from abc import abstractmethod
from typing import List
from numpy import random
from typing_extensions import Self

from simulator.models.event_info import EventInfo


class BaseSimulationEnvironment:
    @abstractmethod
    def __init__(self):
        pass

    def load_data_until(self, final_time: datetime) -> Self:
        self._final_time = final_time
        self._load_data_until(final_time)
        return self

    def set_time(self, time: datetime) -> Self:
        self._time = time
        return self

    def set_rng(self, rng: random.Generator) -> Self:
        self._rng = rng
        return self

    def take_step(self, time_step: timedelta) -> List[EventInfo]:
        self._time += time_step
        return self._take_step(time_step)

    def get_time(self) -> datetime:
        return self._time

    @abstractmethod
    def _take_step(self, time_step: timedelta) -> List[EventInfo]:
        pass

    @abstractmethod
    def _load_data_until(self, final_time: datetime) -> None:
        pass
