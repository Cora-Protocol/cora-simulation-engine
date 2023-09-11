from datetime import timedelta
from abc import abstractmethod
from typing import List

from simulator.models.event_info import EventInfo


class BaseProtocol:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def take_step(self, time_step: timedelta) -> List[EventInfo]:
        pass
