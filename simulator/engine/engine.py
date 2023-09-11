from datetime import datetime, timedelta
from itertools import count
from logging import getLogger
from typing import List, Optional
from uuid import uuid4

from numpy import random
from simulator.environment import BaseSimulationEnvironment
from simulator.metrics import BaseSimulationMetrics
from simulator.models.action_info import ActionInfo
from simulator.models.event_info import EventInfo
from simulator.result import (
    SimulationResult,
    SimulationResultConfig,
    SimulationResultWriter,
)
from simulator.state import SimulationState
from simulator.strategy import BaseSimulationStrategy

logger = getLogger(__name__)


class SimulationEngine:
    def __init__(
        self,
        strategy: BaseSimulationStrategy,
        environment: BaseSimulationEnvironment,
        metrics: BaseSimulationMetrics,
        name: Optional[str] = "simulation",
        config: SimulationResultConfig = SimulationResultConfig(),
    ):
        self._strategy = strategy
        self._environment = environment
        self._metrics = metrics
        self._config = config

        if name and "-" in name:
            raise ValueError("Simulation name cannot contain '-', use '_' instead")

        self.creation_time = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.name = name
        self.hex = uuid4().hex[:7]
        self.engine_id = f"{self.creation_time}-{self.name}-{self.hex}"

        self.run_counter = count()

    def run_simulation(
        self,
        start: datetime,
        end: datetime,
        step: timedelta,
        random_seed: int = 513,  # Actually random from throwing die ;)
    ) -> SimulationResult:

        run_count = next(self.run_counter)
        self._run_id = f"{self.engine_id}-{run_count:06d}"
        logger.info(f"Starting simulation run {self._run_id}...")
        self.result_writer = SimulationResultWriter(self._run_id, self._config)
        rng = random.default_rng(random_seed)

        self._environment.set_time(start)
        self._environment.set_rng(rng)
        self._environment.load_data_until(end)

        self._state = self._get_initial_state(
            start, self._strategy, self._environment, rng
        )
        self._log_step([], [])

        logger.info("Begin simulation run")
        while True:
            self._take_step(step)
            if self._do_stop(end):
                break

        logger.info("Finished simulation run")
        return self.result_writer.get_result()

    def run_simulations(
        self,
        num_runs: int,
        start: datetime,
        end: datetime,
        step: timedelta,
        random_seed: int = 513,
    ) -> List[SimulationResult]:

        random_seeds = [random_seed * (i + 1) for i in range(num_runs)]
        results: List[SimulationResult] = []
        for seed in random_seeds:
            result = self.run_simulation(start, end, step, seed)
            results.append(result)

        return results

    def _get_initial_state(
        self,
        start: datetime,
        strategy: BaseSimulationStrategy,
        environment: BaseSimulationEnvironment,
        rng: random.Generator,
    ) -> SimulationState:
        return SimulationState(start, strategy, environment, rng)

    def _take_step(self, step_size: timedelta) -> None:
        """Run one tick, updates self.state"""
        logger.debug("=====================================")
        logger.debug(f"tick={self._state.get_tick()}: begin")

        actions_info, events_info = self._state.take_step(step_size)
        self._log_step(actions_info, events_info)

        logger.debug(f"tick={self._state.get_tick()}: done")

    def _do_stop(self, stop_datetime: datetime) -> bool:
        if self._state.get_datetime() >= stop_datetime:
            logger.info(f"End simulation: tick={self._state.get_tick()}")
            self._log_end()
            return True
        return False

    def _log_step(
        self, actions_info: List[ActionInfo], events_info: List[EventInfo]
    ) -> None:
        tick = self._state.get_tick()
        timestamp = self._state.get_datetime().strftime("%Y-%m-%d %H:%M:%S")
        step_metrics = self._metrics.by_step(self._state)
        step_metrics = {"tick": tick, "timestamp": timestamp, **step_metrics}
        self.result_writer.add_step_metrics(step_metrics)

        custom_event_metrics = self._metrics.custom_events(
            self._state, actions_info, events_info
        )
        for event_name, metrics in custom_event_metrics.items():
            for metric in metrics:
                self.result_writer.add_custom_event_metrics(event_name, metric)
        for action_info in actions_info:
            self.result_writer.add_action_to_log(action_info)
        for event_info in events_info:
            self.result_writer.add_event_to_log(event_info)

    def _log_end(self) -> None:
        """Log the end of the simulation"""
        all_step_metrics = self.result_writer.result.step_metrics
        end_metrics = self._metrics.end_of_simulation(self._state, all_step_metrics)
        self.result_writer.add_end_metrics(end_metrics)
