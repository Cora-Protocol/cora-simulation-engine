from datetime import datetime, timedelta
from typing import Iterable, List, Literal, NamedTuple, Tuple

import pandas as pd
from protocols.cora.v1.environments import HistoricalCoraEnvironment
from protocols.cora.v1.metrics import CoraMetrics
from protocols.cora.v1.strategies import CoraV1Strategy
from simulator.engine.engine import SimulationEngine, SimulationResultConfig
from pathlib import Path
import zipfile
from itertools import product

LENDING_AMOUNT = 1000_000
GENESIS_PERIOD = timedelta(days=1)
RUNNING_PERIOD = timedelta(days=30)
UPDATE_INTERVAL = timedelta(days=1)

NUM_RANDOMS = 60
UTILIZATION_PARAM_ARRRAY = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
LOAN_START_OPTIONS = ["uniform", "triangular", "parabolic"]
LOAN_DURATION_OPTIONS = ["uniform", "triangular", "parabolic"]
FIRST_BACKTESTING_TIME = datetime(2020, 9, 15)
LAST_BACKTESTING_TIME = datetime(2022, 7, 1)

WORST_MONTHS = [
    datetime(2020, 2, 16),
    datetime(2021, 5, 11),
    datetime(2021, 12, 26),
    datetime(2022, 5, 18),
]

HISTORICAL_ENVIRONMENT = HistoricalCoraEnvironment("ETH")
METRICS = CoraMetrics()

BASE_STRATEGY_DEFINITION = {
    "loan_size_dist": {
        "type": "dist",
        "name": "truncated_lognormal",
        "params": {"lower": 1000, "upper": 100000, "mean": 3.5, "std": 7, "base": 10},
    },
    "ltv_dist": {
        "type": "dist",
        "name": "truncated_normal",
        "params": {"lower": 0.01, "upper": 0.99, "mean": 0.5, "std": 1.11},
    },
    "max_liquidity": LENDING_AMOUNT,
    "initial_lending_amount": LENDING_AMOUNT,
    "genesis_period_seconds": int(GENESIS_PERIOD.total_seconds()),
    "running_period_seconds": int(RUNNING_PERIOD.total_seconds()),
    "fee_model_update_interval_seconds": int(UPDATE_INTERVAL.total_seconds()),
}

CONFIG = SimulationResultConfig(
    results_folder="studies/003-backtesting/results",
    write_custom_event_metrics=False,
    write_step_metrics=False,
    write_end_metrics=False,
    write_log=False,
    meta_log_level="TEST",
)


KELLY_LTV_VALS = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
]

BSM_PARMS = {"lookback_days": 365, "volatility_factor": 1.0}
KELLY_PARAMS = {
    "lookback_days": 365,
    "ltv_values": KELLY_LTV_VALS,
    "max_expiration_days": 30,
    "interval_days": 1,
}
AAVE_PARAMS = {
    "optimal_utilization": 0.8,
    "base_rate": 0.01,
    "rate_slope_1": 0.005,
    "rate_slope_2": 0.75,
}
TRADITIONAL_PARAMS = {
    "optimal_utilization": 1.0,
    "base_rate": 0.01,
    "rate_slope_1": 0.005,
    "rate_slope_2": 0.75,
}

FEE_MODEL_PARAMS = {
    "bsm": {
        "fee_model": "black_scholes",
        "fee_model_update_params": BSM_PARMS,
    },
    "kelly": {
        "fee_model": "cached_kelly",
        "fee_model_update_params": KELLY_PARAMS,
    },
    "aave": {
        "fee_model": "traditional",
        "fee_model_update_params": AAVE_PARAMS,
    },
    "trad": {
        "fee_model": "traditional",
        "fee_model_update_params": TRADITIONAL_PARAMS,
    },
    "bsmaavecombo": {
        "fee_model": "combined_bsm_traditional",
        "fee_model_update_params": {**BSM_PARMS, **AAVE_PARAMS},
    },
    "bsmaavesum": {
        "fee_model": "sum_bsm_traditional",
        "fee_model_update_params": {**BSM_PARMS, **AAVE_PARAMS},
    },
    "bsmtradcombo": {
        "fee_model": "combined_bsm_traditional",
        "fee_model_update_params": {**BSM_PARMS, **TRADITIONAL_PARAMS},
    },
    "bsmtradsum": {
        "fee_model": "sum_bsm_traditional",
        "fee_model_update_params": {**BSM_PARMS, **TRADITIONAL_PARAMS},
    },
    "kellytradcombo": {
        "fee_model": "combined_kelly_traditional",
        "fee_model_update_params": {**KELLY_PARAMS, **TRADITIONAL_PARAMS},
    },
    "kellytradsum": {
        "fee_model": "sum_kelly_traditional",
        "fee_model_update_params": {**KELLY_PARAMS, **TRADITIONAL_PARAMS},
    },
}


DistOptions = Literal["uniform", "triangular", "parabolic"]


class BacktestingPoint(NamedTuple):
    utilization_parameter: float
    loan_start_type: DistOptions
    loan_duration_type: DistOptions


def get_strategy(
    ltv: float,
    fee_model: str,
    fee_model_update_params: dict,
    utilization_parameter: float,
    loan_start_type: DistOptions,
    loan_duration_type: DistOptions,
):
    strategy_definition = BASE_STRATEGY_DEFINITION.copy()
    strategy_definition["max_ltv"] = ltv
    strategy_definition["fee_model"] = fee_model
    strategy_definition["fee_model_update_params"] = fee_model_update_params
    strategy_definition["utilization_parameter"] = utilization_parameter
    strategy_definition["loan_start_dist"] = {
        "type": "dist",
        "name": loan_start_type,
        "params": {"lower": 0, "upper": 1}
        if loan_start_type == "uniform"
        else {"lower": 1, "upper": 0},
    }
    strategy_definition["loan_duration_dist"] = {
        "type": "dist",
        "name": loan_duration_type,
        "params": {"lower": 0, "upper": 1},
    }
    return CoraV1Strategy.from_dict(strategy_definition)


def iterate_backtesting_points(
    utilization_param_array: List[float],
    loan_start_type_options: List[DistOptions] = LOAN_START_OPTIONS,
    loan_duration_type_options: List[DistOptions] = LOAN_DURATION_OPTIONS,
) -> Iterable[BacktestingPoint]:
    for utilization_parameter in utilization_param_array:
        for loan_start_type in loan_start_type_options:
            for loan_duration_type in loan_duration_type_options:
                yield BacktestingPoint(
                    utilization_parameter, loan_start_type, loan_duration_type
                )


def generate_study(
    ltv: float,
    fee_model: str,
    fee_model_update_params: dict,
    utilization_param_array: List[float] = UTILIZATION_PARAM_ARRRAY,
    loan_start_type_options: List[DistOptions] = LOAN_START_OPTIONS,
    loan_duration_type_options: List[DistOptions] = LOAN_DURATION_OPTIONS,
    start_times: List[datetime] = WORST_MONTHS,
    num_randoms: int = 20,
    random_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_sim_time = GENESIS_PERIOD + RUNNING_PERIOD + timedelta(hours=1)

    num_sims = (
        num_randoms
        * len(start_times)
        * len(utilization_param_array)
        * len(loan_start_type_options)
        * len(loan_duration_type_options)
    )

    aprox_time_mins = (num_sims * 1.5) // 60
    aprox_time_secs = (num_sims * 1.5) % 60
    print(
        f"Running {num_sims} simulations, approximate time {aprox_time_mins} minues and {aprox_time_secs} seconds."
    )

    df_input = []
    timeseries_dfs = []
    for backtesting_point in iterate_backtesting_points(
        utilization_param_array, loan_start_type_options, loan_duration_type_options
    ):
        strategy = get_strategy(
            ltv,
            fee_model,
            fee_model_update_params,
            backtesting_point.utilization_parameter,
            backtesting_point.loan_start_type,
            backtesting_point.loan_duration_type,
        )
        engine = SimulationEngine(
            strategy, HISTORICAL_ENVIRONMENT, METRICS, config=CONFIG
        )
        for i, start_time in enumerate(start_times):
            for j, random_seed in enumerate(
                range(random_seed, random_seed + num_randoms)
            ):
                result = engine.run_simulation(
                    start_time,
                    start_time + total_sim_time,
                    timedelta(hours=1),
                    random_seed,
                )
                custom_metrics = result.custom_event_metrics["cycle_end"][0]

                df_input.append(
                    {
                        "max_ltv": ltv,
                        "fee_model": fee_model,
                        "utilization_parameter": backtesting_point.utilization_parameter,
                        "loan_start_type": backtesting_point.loan_start_type,
                        "loan_duration_type": backtesting_point.loan_duration_type,
                        "start_time": start_time,
                        "seed": random_seed,
                        **custom_metrics,
                    }
                )

                step_metrics = pd.DataFrame(result.step_metrics)
                step_metrics = step_metrics[
                    [
                        "timestamp",
                        "collateral_price",
                        "pool_utilization",
                        "pool_liquid_capital",
                        "pool_capital_lent",
                        "active_loans_count",
                        "defaulted_loans_count",
                        "paid_loans_count",
                        "expired_loans_count",
                        "total_loans_count",
                        "reclaimed_collateral",
                        "active_loan_collateral",
                        "active_loans_amount",
                        "collateral_ratio",
                        "pool_realized_pnl",
                        "pool_unrealized_pnl",
                        "borrows",
                        "earned_fees",
                        "sum_of_fees",
                    ]
                ]
                step_metrics["max_ltv"] = ltv
                step_metrics["fee_model"] = fee_model
                step_metrics[
                    "utilization_parameter"
                ] = backtesting_point.utilization_parameter
                step_metrics["loan_start_type"] = backtesting_point.loan_start_type
                step_metrics[
                    "loan_duration_type"
                ] = backtesting_point.loan_duration_type
                step_metrics["start_time"] = start_time
                step_metrics["seed"] = random_seed
                timeseries_dfs.append(step_metrics)

    metrics_df = pd.DataFrame(df_input)
    tiemseries_df = pd.concat(timeseries_dfs, ignore_index=True)
    return metrics_df, tiemseries_df


def float_to_name(number: float) -> str:
    return f"{number:.2f}".replace(".", "_")


def run_and_save_study(
    save_location: Path,
    max_ltv: float,
    utilization_param: float,
    fee_model: Literal[
        "bsm",
        "kelly",
        "aave",
        "trad",
        "bsmaavecombo",
        "bsmaavesum",
        "bsmtradcombo",
        "bsmtradsum",
        "kellytradcombo",
        "kellytradsum",
    ],
    loan_start_type: str,
    loan_duration_type: str,
    num_randoms: int,
    random_seed: int = 0,
    cancel_if_exists: bool = True,
) -> Tuple[Path, Path]:
    csv_filename = f"blackswan_{fee_model}_ltv{float_to_name(max_ltv)}_up{float_to_name(utilization_param)}_x{num_randoms}.csv"
    csv_file_location = save_location / csv_filename
    parquet_file_location = csv_file_location.with_suffix(".parquet")
    if (
        cancel_if_exists
        and csv_file_location.exists()
        and parquet_file_location.exists()
    ):
        print(f"{csv_filename} already exists, skipping.")
        return csv_file_location, parquet_file_location

    print(f"Running to get {csv_filename}")
    study_results, timeseries = generate_study(
        ltv=max_ltv,
        fee_model=FEE_MODEL_PARAMS[fee_model]["fee_model"],
        fee_model_update_params=FEE_MODEL_PARAMS[fee_model]["fee_model_update_params"],
        utilization_param_array=[utilization_param],
        loan_start_type_options=[loan_start_type],
        loan_duration_type_options=[loan_duration_type],
        start_times=WORST_MONTHS,
        num_randoms=num_randoms,
        random_seed=random_seed,
    )
    study_results.to_csv(csv_file_location)
    timeseries.to_parquet(parquet_file_location)
    print(f"Saved {csv_filename} and {parquet_file_location}")
    return csv_file_location, parquet_file_location


def run_and_save_study_array(
    save_location: Path,
    max_ltvs: List[float],
    utilization_params: List[float],
    fee_model: Literal[
        "bsm",
        "kelly",
        "aave",
        "trad",
        "bsmaavecombo",
        "bsmaavesum",
        "bsmtradcombo",
        "bsmtradsum",
        "kellytradcombo",
        "kellytradsum",
    ],
    loan_start_type: str,
    loan_duration_type: str,
    num_randoms: int = NUM_RANDOMS,
    random_seed: int = 0,
    cancel_if_exists: bool = True,
    continue_if_failed: bool = True,
    compress_to_zip: bool = True,
):
    save_location = Path(save_location)
    save_location.mkdir(exist_ok=True, parents=True)
    paths: List[Path] = []
    for i, (max_ltv, up) in enumerate(product(max_ltvs, utilization_params)):
        if continue_if_failed:
            try:
                csv_path, parquet_path = run_and_save_study(
                    save_location,
                    max_ltv,
                    up,
                    fee_model,
                    loan_start_type,
                    loan_duration_type,
                    num_randoms,
                    random_seed + i,
                    cancel_if_exists,
                )
            except Exception as e:
                print(
                    f"Failed to run {max_ltv} {up} {fee_model} {loan_start_type} {loan_duration_type}"
                )
                print(e)
        else:
            csv_path, parquet_path = run_and_save_study(
                save_location,
                max_ltv,
                up,
                fee_model,
                loan_start_type,
                loan_duration_type,
                num_randoms,
                random_seed + i,
                cancel_if_exists,
            )
        paths.append(csv_path)
        paths.append(parquet_path)

    if compress_to_zip:
        zip_filename = f"00_blackswan_{fee_model}_x{num_randoms}.zip"
        zip_file_location = save_location / zip_filename
        with zipfile.ZipFile(zip_file_location, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file in paths:
                zip_file.write(file, file.name)
        print(f"Compressed {zip_filename}")
