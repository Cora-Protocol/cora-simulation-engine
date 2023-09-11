from datetime import datetime, timedelta
import pathlib
from unittest import TestCase
from protocols.cora.v1.environments import HistoricalCoraEnvironment
from protocols.cora.v1.metrics import CoraMetrics
from protocols.cora.v1.strategies import (
    CoraV1Strategy,
)
from simulator.engine.engine import SimulationEngine, SimulationResultConfig


class V1Strategy(TestCase):
    def test_strategy(self):
        file_location = pathlib.Path(__file__).parent.absolute() / "test_1_params.json"
        strategy = CoraV1Strategy.from_json_file(file_location)
        environment = HistoricalCoraEnvironment("ETH")
        metrics = CoraMetrics()

        simulator_engine = SimulationEngine(
            strategy=strategy,
            environment=environment,
            metrics=metrics,
            config=SimulationResultConfig(
                write_step_metrics=False,
                write_end_metrics=False,
                write_log=False,
                write_custom_event_metrics=False,
                meta_log_level="TEST",
            ),
        )

        simulator_engine.run_simulation(
            start=datetime(2022, 1, 1),
            end=datetime(2022, 1, 1) + timedelta(days=60),
            step=timedelta(hours=1),
        )
