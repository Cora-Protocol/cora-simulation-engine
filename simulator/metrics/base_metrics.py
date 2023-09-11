from abc import abstractmethod
from typing import Dict, List, Union

from simulator.models.action_info import ActionInfo
from simulator.models.event_info import EventInfo
from simulator.state import SimulationState

Metric = Dict[str, Union[str, int, float]]


class BaseSimulationMetrics:
    @classmethod
    @abstractmethod
    def by_step(cls, state: SimulationState) -> Metric:
        pass

    @classmethod
    @abstractmethod
    def custom_events(
        cls,
        state: SimulationState,
        actions_info: List[ActionInfo],
        events_info: List[EventInfo],
    ) -> Dict[str, List[dict]]:
        pass

    @classmethod
    @abstractmethod
    def end_of_simulation(cls, state: SimulationState, metrics: List[Metric]) -> Metric:
        pass
