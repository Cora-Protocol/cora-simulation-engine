from abc import abstractmethod
from typing import List

from simulator.environment.base_environment import BaseSimulationEnvironment
from ..models.action_info import ActionInfo
from simulator.protocol.base_protocol import BaseProtocol


class BaseAgent:
    _priority = 0

    def __init__(
        self, id: str, protocol: BaseProtocol, environment: BaseSimulationEnvironment
    ):
        self.id = id
        self.protocol = protocol
        self.environment = environment

    @abstractmethod
    def act() -> List[ActionInfo]:
        pass

    @abstractmethod
    def get_priority(self) -> int:
        return self._priority
