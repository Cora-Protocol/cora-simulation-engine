from datetime import datetime, timedelta
from random import random
from simulator.environment import BaseSimulationEnvironment
from simulator.models.action_info import ActionInfo
from simulator.models.event_info import EventInfo
from simulator.strategy import BaseSimulationStrategy
from simulator.agent import BaseAgent
from typing import List, Tuple
import numpy as np
from numpy import random


class SimulationState:
    def __init__(
        self,
        time: datetime,
        strategy: BaseSimulationStrategy,
        environment: BaseSimulationEnvironment,
        rng: random.Generator,
    ):
        self._rng = rng

        self._tick = 0
        self._time = time

        self._strategy = strategy
        self._strategy.set_rng(self._rng)

        self._environment = environment

        # The simulation strategy determines the initial state of protocol and agents.
        self._protocol = self._strategy.get_initial_protocol(self._environment)
        self._agents = self._strategy.get_initial_agents(
            self._environment, self._protocol
        )

    def take_step(
        self, time_step: timedelta
    ) -> Tuple[List[ActionInfo], List[EventInfo]]:
        self._tick += 1
        self._time += time_step

        # The environment advances, updating time and prices
        environment_events_info = self._environment.take_step(time_step)

        # The protocol advances, some automatic actions might take place
        protocol_events_info = self._protocol.take_step(time_step)

        # The strategy might introduce or eliminate agents from the simulation.
        self._agents = self._strategy.update_agents(
            self._agents, self._protocol, self._environment, time_step
        )

        # Agents act depending on the protocol state and environment, modifying them.
        actions_info: List[ActionInfo] = []
        for agent in self._agents_by_priority():
            actions_info.extend(agent.act())  # TODO: agent.act()
        return actions_info, environment_events_info + protocol_events_info

    def get_datetime(self) -> datetime:
        return self._time

    def get_tick(self) -> int:
        return self._tick

    def _agents_by_priority(self) -> List[BaseAgent]:
        if self._agents is None:
            return []

        return sorted(self._agents, key=lambda agent: agent.get_priority())
