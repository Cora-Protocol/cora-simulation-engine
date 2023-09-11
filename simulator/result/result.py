from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from simulator.models.action_info import ActionInfo
from simulator.models.event_info import EventInfo
from simulator.utilities.data_storage import DataStorage
from simulator.utilities.logger import logger


@dataclass
class SimulationResult:
    run_id: str
    step_metrics: List[dict] = field(default_factory=list)
    custom_event_metrics: Dict[str, List[dict]] = field(default_factory=dict)
    end_metrics: dict = field(default_factory=dict)
    action_log: List[dict] = field(default_factory=list)
    event_log: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationResultConfig:
    results_folder: str = "simlogs"
    write_step_metrics: bool = True
    write_end_metrics: bool = True
    write_custom_event_metrics: bool = True
    write_log: bool = True
    meta_log_level: str = "INFO"
    step_log_interval: int = 1


class SimulationResultWriter:
    def __init__(self, run_id: str, config: SimulationResultConfig):
        self.result = SimulationResult(run_id)
        self.config = config

        logger.set_meta_log_level(config.meta_log_level)
        logger.set_write_actions(config.write_log)
        logger.set_write_events(config.write_log)

        if (
            config.write_log
            or config.write_custom_event_metrics
            or config.write_end_metrics
            or config.write_step_metrics
        ):
            files_location = Path(config.results_folder) / run_id
            files_location.mkdir(parents=True, exist_ok=True)
            self.storage = DataStorage(files_location)
            logger.set_filename(files_location / "log.txt")

    def get_result(self) -> SimulationResult:
        return self.result

    def add_step_metrics(self, step_metrics: dict):
        self.result.step_metrics.append(step_metrics)
        if self.config.write_step_metrics:
            self.storage.write_or_append_csv("step_metrics", [step_metrics])

    def add_custom_event_metrics(self, event_name: str, event_metrics: dict):
        if event_name not in self.result.custom_event_metrics:
            self.result.custom_event_metrics[event_name] = []
        self.result.custom_event_metrics[event_name].append(event_metrics)
        if self.config.write_custom_event_metrics:
            self.storage.write_or_append_csv(event_name, [event_metrics])

    def add_end_metrics(self, end_metrics: dict):
        self.result.end_metrics = end_metrics
        if self.config.write_end_metrics:  # TODO: JSONify
            self.storage.write_or_append_csv("end_metrics", [end_metrics])

    def add_action_to_log(self, action_info: ActionInfo):
        self.result.action_log.append(action_info)
        if self.config.write_log:
            logger.log_action(action_info)

    def add_event_to_log(self, event_info: EventInfo):
        self.result.event_log.append(event_info)
        if self.config.write_log:
            logger.log_event(event_info)
