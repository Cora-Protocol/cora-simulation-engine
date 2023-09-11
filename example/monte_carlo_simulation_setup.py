import zipfile
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def graph_ltv_var(d: dict, max_var: float, max_ltv: float):

    var_vals = list(d.values())
    ltv_vals = list(d.keys())

    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(ltv_vals, var_vals, "o-")
    ax.axhline(max_var, color="red", linestyle="--")
    ax.scatter([max_ltv], [d[max_ltv]], color="red", marker="o", s=100)

    ax.set_xlabel("LTV", fontsize=14)
    ax.set_ylabel("Monthly VAR 95%", fontsize=14)
    ax.set_title("LTV vs. Monthly VAR", fontsize=16)
    plt.show()


def run_study(
    max_ltv_options: List[float],
    fee_model: str,
    borrower_demand_options: List[float],
    asset: str = "ETH",
    loan_start_options: List[DistOptions] = ["parabolic"],
    loan_duration_options: List[DistOptions] = ["parabolic"],
    volatility_factor_options: List[float] = [1.0, 1.5, 2.0],
    fee_model_update_params: Optional[dict] = None,
    zero_mu: bool = True,
    start_time: datetime = datetime(2022, 9, 14),
    num_seeds: int = 100,
    pass_if_exists: bool = True,
    compress_to_zip: bool = False,
    save_folder: str = "example/outputs",
    base_seed: int = 95739,
    return_paths: bool = False,
    verbose: bool = True,
):

    if fee_model in FEE_MODEL_PARAMS:
        fee_model_name = FEE_MODEL_PARAMS[fee_model]["fee_model"]
        fee_model_params = FEE_MODEL_PARAMS[fee_model]["fee_model_update_params"]
        if fee_model_update_params is not None:
            fee_model_params = fee_model_update_params

    else:
        fee_model_name = fee_model
        fee_model_params = fee_model_update_params
        if fee_model_params is None:
            raise ValueError(
                f"Must provide fee_model_update_params when fee_model is not in the default set: {FEE_MODEL_PARAMS.keys()}."
            )

    total_sim_time = GENESIS_PERIOD + RUNNING_PERIOD + timedelta(hours=1)
    total_simulations = (
        len(max_ltv_options)
        * len(borrower_demand_options)
        * len(loan_start_options)
        * len(loan_duration_options)
        * len(volatility_factor_options)
        * num_seeds
    )
    if verbose:
        print(
            f"Total simulations: {total_simulations}. Estimated time: {1.5 * total_simulations / 60:.0f} minutes."
        )

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
        if verbose:
            print(
                f"{ii + 1}/{total_study_variations} RUNNING studies for {volatility_factor=}, {borrower_demand=}, {loan_start=}, {loan_duration=}"
            )
        subfolder_name = f"vol_{float_to_str(volatility_factor)}_demand_{float_to_str(borrower_demand)}_start_{loan_start}_duration_{loan_duration}"
        subfolder = Path(save_folder) / subfolder_name
        subfolder.mkdir(parents=True, exist_ok=True)

        environment = BrownianCoraEnvironment(
            asset, volatility_factor=volatility_factor, zero_mu=zero_mu
        )
        num_studies = len(max_ltv_options)
        paths: List[Path] = []
        for jj, max_ltv in enumerate(max_ltv_options):

            if verbose:
                print(
                    f"{jj+1}/{num_studies}: Running study for {max_ltv=}, {fee_model=} with {num_seeds} seeds"
                )
            strategy_name = f"cora_{asset}_maxltv_{float_to_str(max_ltv)}_feemodel_{fee_model}_n_{num_seeds}_s_{seed}"
            csv_filepath = subfolder / f"{strategy_name}.csv"
            paths.append(csv_filepath)

            if csv_filepath.exists() and pass_if_exists:
                if verbose:
                    print(
                        f"Skipping {strategy_name} becuase {csv_filepath} already exists"
                    )
                continue

            strategy = get_strategy(
                max_ltv=max_ltv,
                fee_model=fee_model_name,
                fee_model_update_params=fee_model_params,
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
            if verbose:
                print(f"Saved {csv_filepath}")

        if compress_to_zip:
            zip_file_location = Path(save_folder) / f"{subfolder_name}.zip"
            with zipfile.ZipFile(
                zip_file_location, "w", zipfile.ZIP_DEFLATED
            ) as zip_file:
                for file in paths:
                    zip_file.write(file, file.name)
            if verbose:
                print(f"Compressed {zip_file_location.name}")

    if return_paths:
        return paths


def find_ltv_for_var(
    monthly_var_ratio_limit: float,
    asset: str = "ETH",
    fee_model: str = "bsmtradsum",
    borrower_demand: float = 0.5,
    max_ltv_options: List[float] = [
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
    ],
    loan_start: DistOptions = "parabolic",
    loan_duration: DistOptions = "parabolic",
    volatility_factor: float = 1.0,
    fee_model_update_params: Optional[Dict[str, Any]] = None,
    zero_mu: bool = True,
    start_time: datetime = datetime(2022, 9, 14),
    num_seeds: int = 100,
    pass_if_exists: bool = True,
    compress_to_zip: bool = False,
    save_folder: str = "example/outputs",
    base_seed: int = 95739,
) -> float:

    if monthly_var_ratio_limit <= 0:
        raise ValueError("monthly_var_ratio_limit must be positive")

    paths = {}
    for max_ltv in max_ltv_options:
        paths[max_ltv] = run_study(
            fee_model=fee_model,
            max_ltv_options=[max_ltv],
            asset=asset,
            borrower_demand_options=[borrower_demand],
            loan_start_options=[loan_start],
            loan_duration_options=[loan_duration],
            volatility_factor_options=[volatility_factor],
            fee_model_update_params=fee_model_update_params,
            zero_mu=zero_mu,
            start_time=start_time,
            num_seeds=num_seeds,
            pass_if_exists=pass_if_exists,
            compress_to_zip=compress_to_zip,
            save_folder=save_folder,
            base_seed=base_seed,
            return_paths=True,
            verbose=False,
        )[0]

    dfs: Dict[float, pd.DataFrame] = {}
    for max_ltv, path in paths.items():
        df = pd.read_csv(path)
        dfs[max_ltv] = df

    var_values: Dict[float, float] = {}
    for max_ltv, df in dfs.items():
        var_values[max_ltv] = (-df["pnl_ratio"]).quantile(0.95)

    # select the max LTV that has a var less than the limit
    for max_ltv in sorted(list(var_values.keys()), reverse=True):
        if var_values[max_ltv] < monthly_var_ratio_limit:
            selected_max_ltv = max_ltv
            break
    # if none of the LTVs are below the limit, return the lowest LTV
    else:
        selected_max_ltv = min(var_values.keys())

    var_values = {k: max(v, 0.0) for k, v in var_values.items()}

    graph_ltv_var(var_values, monthly_var_ratio_limit, selected_max_ltv)

    return selected_max_ltv
