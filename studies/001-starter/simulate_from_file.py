from pathlib import Path
from simulator.engine.engine import SimulationResultConfig, SimulationEngine
from protocols.cora.v1.environments import HistoricalCoraEnvironment
from protocols.cora.v1.metrics import CoraMetrics
from protocols.cora.v1.strategies import CoraV1Strategy
from datetime import datetime, timedelta

STRATEGY_FILE = Path("studies/001-starter/strategy_example.json")


if __name__ == "__main__":

    strategy = CoraV1Strategy.from_json_file(STRATEGY_FILE)
    environment = HistoricalCoraEnvironment(symbol="ETH")
    metrics = CoraMetrics()
    config = SimulationResultConfig(
        results_folder="studies/001-starter/results",
        write_step_metrics=True,
        write_end_metrics=True,
        write_log=True,
        write_custom_event_metrics=True,
        meta_log_level="INFO",
    )

    simulation_engine = SimulationEngine(
        strategy=strategy,
        environment=environment,
        metrics=metrics,
        name="starter",
        config=config,
    )

    start_date = datetime(2022, 1, 1)
    end_date = start_date + timedelta(days=31, hours=1)
    time_step = timedelta(hours=1)

    print("Simulating...")
    simulation_engine.run_simulation(
        start=start_date,
        end=end_date,
        step=time_step,
        random_seed=345,
    )

    print(f"Simulation complete => {simulation_engine._run_id}")
