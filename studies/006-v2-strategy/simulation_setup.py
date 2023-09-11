import zipfile
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Iterable, List, Literal, NamedTuple, Tuple

import pandas as pd
from protocols.cora.v1.environments import BrownianCoraEnvironment
from protocols.cora.v1.metrics import CoraMetrics
from protocols.cora.v1.strategies import CoraV2Strategy
from simulator.engine.engine import SimulationEngine, SimulationResultConfig

LENDING_AMOUNT = 1000_000
GENESIS_PERIOD = timedelta(days=1)
RUNNING_PERIOD = timedelta(days=30)
UPDATE_INTERVAL = timedelta(days=1)

UTILIZATION_PARAM_ARRRAY = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
LOAN_START_OPTIONS = ["uniform", "triangular", "parabolic"]
LOAN_DURATION_OPTIONS = ["uniform", "triangular", "parabolic"]
BACKTESTING_START_TIME = datetime(2022, 8, 24)

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
    results_folder="studies/006-v2-strategy/results",
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


def get_strategy(
    max_ltv: float,
    fee_model: str,
    fee_model_update_params: dict,
    borrower_demand_ratio: float,
    loan_start_type: DistOptions,
    loan_duration_type: DistOptions,
):
    strategy_definition = BASE_STRATEGY_DEFINITION.copy()
    strategy_definition["max_ltv"] = max_ltv
    strategy_definition["fee_model"] = fee_model
    strategy_definition["fee_model_update_params"] = fee_model_update_params
    strategy_definition["borrower_demand_ratio"] = borrower_demand_ratio
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
    return CoraV2Strategy.from_dict(strategy_definition)


def float_to_str(x):
    return f"{x:.2f}".replace(".", "_")


def run_study(
    max_ltv_options: List[float],
    fee_model_options: List[str],
    borrower_demand_options: List[float],
    loan_start_options: List[DistOptions] = ["uniform", "triangular", "parabolic"],
    loan_duration_options: List[DistOptions] = ["uniform", "triangular", "parabolic"],
    volatility_factor_options: List[float] = [1.0, 1.5, 2.0],
    zero_mu: bool = True,
    start_time: datetime = datetime(2022, 9, 14),
    num_seeds: int = 100,
    pass_if_exists: bool = True,
    compress_to_zip: bool = True,
    save_folder: str = "studies/006-v2-strategy/results",
    base_seed: int = 95739,
):
    total_sim_time = GENESIS_PERIOD + RUNNING_PERIOD + timedelta(hours=1)
    total_simulations = (
        len(max_ltv_options)
        * len(fee_model_options)
        * len(borrower_demand_options)
        * len(loan_start_options)
        * len(loan_duration_options)
        * len(volatility_factor_options)
        * num_seeds
    )
    print(f"Total simulations: {total_simulations}. Let's burn that CPU!")

    total_study_variations = (
        len(borrower_demand_options)
        * len(loan_start_options)
        * len(loan_duration_options)
        * len(volatility_factor_options)
    )

    seed = base_seed
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    for ii, (
        volatility_factor,
        borrower_demand,
        loan_start,
        loan_duration,
    ) in enumerate(
        product(
            volatility_factor_options,
            borrower_demand_options,
            loan_start_options,
            loan_duration_options,
        )
    ):
        print(
            f"{ii + 1}/{total_study_variations} RUNNING studies for {volatility_factor=}, {borrower_demand=}, {loan_start=}, {loan_duration=}"
        )
        subfolder_name = f"vol_{float_to_str(volatility_factor)}_demand_{float_to_str(borrower_demand)}_start_{loan_start}_duration_{loan_duration}"
        subfolder = Path(save_folder) / subfolder_name
        subfolder.mkdir(parents=True, exist_ok=True)

        environment = BrownianCoraEnvironment(
            "ETH", volatility_factor=volatility_factor, zero_mu=zero_mu
        )
        num_studies = len(max_ltv_options) * len(fee_model_options)
        paths: List[Path] = []
        for jj, (max_ltv, fee_model) in enumerate(
            product(max_ltv_options, fee_model_options)
        ):
            print(
                f"{jj+1}/{num_studies}: Running study for {max_ltv=}, {fee_model=} with {num_seeds} seeds"
            )
            strategy_name = f"cora_maxltv_{float_to_str(max_ltv)}_feemodel_{fee_model}_n_{num_seeds}_s_{seed}"
            csv_filepath = subfolder / f"{strategy_name}.csv"
            paths.append(csv_filepath)

            if csv_filepath.exists() and pass_if_exists:
                print(f"Skipping {strategy_name} becuase {csv_filepath} already exists")
                continue

            strategy = get_strategy(
                max_ltv=max_ltv,
                fee_model=FEE_MODEL_PARAMS[fee_model]["fee_model"],
                fee_model_update_params=FEE_MODEL_PARAMS[fee_model][
                    "fee_model_update_params"
                ],
                borrower_demand_ratio=borrower_demand,
                loan_start_type=loan_start,
                loan_duration_type=loan_duration,
            )
            engine = SimulationEngine(strategy, environment, METRICS, config=CONFIG)
            rows = []
            for kk in range(num_seeds):
                result = engine.run_simulation(
                    start_time,
                    start_time + total_sim_time,
                    timedelta(hours=1),
                    seed,
                )
                custom_metrics = result.custom_event_metrics["cycle_end"][0]
                rows.append(
                    {
                        "max_ltv": max_ltv,
                        "fee_model": fee_model,
                        "borrower_demand": borrower_demand,
                        "loan_start_type": loan_start,
                        "loan_duration_type": loan_duration,
                        "volatility_factor": volatility_factor,
                        "start_time": start_time,
                        "seed": seed,
                        "mc_iteration": kk,
                        **custom_metrics,
                    }
                )
                seed += 1

            df = pd.DataFrame(rows)
            df.to_csv(csv_filepath, index=False)
            print(f"Saved {csv_filepath}")

        if compress_to_zip:
            zip_file_location = Path(save_folder) / f"{subfolder_name}.zip"
            with zipfile.ZipFile(
                zip_file_location, "w", zipfile.ZIP_DEFLATED
            ) as zip_file:
                for file in paths:
                    zip_file.write(file, file.name)
            print(f"Compressed {zip_file_location.name}")
