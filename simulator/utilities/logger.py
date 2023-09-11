import logging
import os
from pathlib import Path
from typing import Optional, Union

from simulator.models.action_info import ActionInfo
from simulator.models.event_info import EventInfo

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()


class SimulationLogger:
    def __init__(self):

        self._level = "INFO"

        self.meta_logger = logging.getLogger(f"meta")
        self.action_logger = logging.getLogger(f"action")
        self.event_logger = logging.getLogger(f"event")

        self.action_logger.setLevel("INFO")
        self.event_logger.setLevel("INFO")

        self._write_actions = True
        self._write_events = True

    def set_meta_log_level(self, level: str):
        self._level = level
        if level == "TEST":
            self.meta_logger.setLevel("CRITICAL")
        else:
            self.meta_logger.setLevel(level)

    def set_write_actions(self, write_actions: bool):
        self._write_actions = write_actions
        self.action_logger.setLevel("INFO" if write_actions else "CRITICAL")
        if not write_actions:
            self.action_logger.handlers = []

    def set_write_events(self, write_events: bool):
        self._write_events = write_events
        self.event_logger.setLevel("INFO" if write_events else "CRITICAL")
        if not write_events:
            self.event_logger.handlers = []

    def set_filename(self, filename: Union[Path, str]):
        self.action_logger.handlers = []
        self.event_logger.handlers = []
        self.action_logger.handlers = []

        if not (self._level == "TEST"):
            self.file_handler = logging.FileHandler(filename)
            self.formatter = logging.Formatter(
                "%(levelname)s::%(name)s::%(message)s::%(extra)s"
            )
            self.file_handler.setFormatter(self.formatter)

            self.meta_logger.addHandler(self.file_handler)
            if self._write_actions:
                self.action_logger.addHandler(self.file_handler)
            if self._write_events:
                self.event_logger.addHandler(self.file_handler)

    def log_action(self, action_info: ActionInfo):
        extra = {
            "agent_id": action_info.agent_id,
            "time": action_info.time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": action_info.type,
            **action_info.extra,
        }
        self.action_logger.info(action_info.message, extra={"extra": extra or {}})

    def log_event(self, event_info: EventInfo):
        extra = {
            "time": event_info.time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_info.type,
            **event_info.extra,
        }
        self.event_logger.info(event_info.message, extra={"extra": extra or {}})

    def info(self, message: str, params: Optional[dict] = None):
        self.meta_logger.info(message, extra={"extra": params or {}})

    def warning(self, message: str, params: Optional[dict] = None):
        self.meta_logger.warning(message, extra={"extra": params or {}})

    def error(self, message: str, params: Optional[dict] = None):
        self.meta_logger.error(message, extra={"extra": params or {}})

    def debug(self, message: str, params: Optional[dict] = None):
        self.meta_logger.debug(message, extra={"extra": params or {}})

    def critical(self, message: str, params: Optional[dict] = None):
        self.meta_logger.critical(message, extra={"extra": params or {}})

    def exception(self, message: str, params: Optional[dict] = None):
        self.meta_logger.exception(message, extra={"extra": params or {}})


logger = SimulationLogger()
