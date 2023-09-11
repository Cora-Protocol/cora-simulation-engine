from abc import abstractmethod
from datetime import datetime
from typing import List

from numpy import random
from simulator.agent import BaseAgent
from simulator.environment import BaseSimulationEnvironment
from simulator.protocol import BaseProtocol
from simulator.strategy.strategy_params_parser import StrategyParamsParser
from simulator.utilities.distributions import BaseDistribution
from typing_extensions import Self


class BaseSimulationStrategy:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @classmethod
    def from_dict(cls, parameters: dict) -> Self:
        parsed_parameters = StrategyParamsParser.parse_strategy_params(parameters)
        return cls(**parsed_parameters)

    @classmethod
    def from_json_file(cls, file_path: str) -> Self:
        parameters = StrategyParamsParser.parse_json(file_path)
        return cls.from_dict(parameters)

    def set_rng(self, rng: random.Generator) -> None:
        self._rng = rng
        for param in self._kwargs.values():
            if isinstance(param, BaseDistribution):
                param.set_rng(rng)

    @abstractmethod
    def get_initial_protocol(
        self, environment: BaseSimulationEnvironment
    ) -> BaseProtocol:
        pass

    @abstractmethod
    def get_initial_agents(
        self, environment: BaseSimulationEnvironment
    ) -> List[BaseAgent]:
        pass

    @abstractmethod
    def update_agents(
        self,
        agents: List[BaseAgent],
        protocol: BaseProtocol,
        environment: BaseSimulationEnvironment,
        time_step: datetime,
    ) -> List[BaseAgent]:
        pass
