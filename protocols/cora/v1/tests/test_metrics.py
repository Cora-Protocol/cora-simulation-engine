import math
import re
from datetime import datetime, timedelta
from itertools import count
from random import randint, random
from typing import Pattern, Union

from protocols.cora.v1.business_logic.lending_pool import Loan
from protocols.cora.v1.environments import HistoricalCoraEnvironment
from protocols.cora.v1.metrics import CoraMetrics
from protocols.cora.v1.strategies import CoraV1Strategy
from simulator.engine.engine import SimulationResultConfig, SimulationEngine
from simulator.strategy.strategy_params_parser import StrategyParamsParser

DEFAULT_STRATEGY_PARAMS = {
    "utilization_parameter": 0.7,
    "loan_size_dist": {
        "type": "dist",
        "name": "truncated_lognormal",
        "params": {"lower": 0, "upper": 100000, "mean": 3, "std": 1, "base": 10},
    },
    "loan_start_dist": {
        "type": "dist",
        "name": "uniform",
        "params": {"lower": 0, "upper": 0.033333},
    },
    "loan_duration_dist": {
        "type": "dist",
        "name": "uniform",
        "params": {"lower": 0.5, "upper": 0.95},
    },
    "ltv_dist": {
        "type": "dist",
        "name": "uniform",
        "params": {"lower": 0.2, "upper": 0.6},
    },
    "max_ltv": 0.9,
    "max_liquidity": 1000000,
    "initial_lending_amount": 10000,
    "genesis_period_seconds": 604800,
    "running_period_seconds": 2592000,
    "fee_model": "black_scholes",
    "fee_model_update_params": {"lookback_days": 365, "volatility_factor": 1.0},
    "fee_model_update_interval_seconds": 86400,
}


def generate_ramdom_strategy_params():
    return {
        "utilization_parameter": random(),
        "loan_size_dist": {
            "type": "dist",
            "name": "truncated_lognormal",
            "params": {
                "lower": randint(0, 10000),
                "upper": randint(10000, 100000),
                "mean": randint(0, 10),
                "std": randint(1, 10),
                "base": 10,
            },
        },
        "loan_start_dist": {
            "type": "dist",
            "name": "uniform",
            "params": {"lower": 0, "upper": random()},
        },
        "loan_duration_dist": {
            "type": "dist",
            "name": "uniform",
            "params": {"lower": 0, "upper": random()},
        },
        "ltv_dist": {
            "type": "dist",
            "name": "uniform",
            "params": {"lower": 0, "upper": random()},
        },
        "max_ltv": random(),
        "max_liquidity": randint(0, 1000000),
        "initial_lending_amount": randint(0, 1000),
        "genesis_period_seconds": randint(0, 604800),
        "running_period_seconds": randint(0, 2592000),
        "fee_model": "black_scholes",
        "fee_model_update_params": {
            "lookback_days": randint(2, 365),
            "volatility_factor": random(),
        },
        "fee_model_update_interval_seconds": randint(0, 86400),
    }


def run_simulation(
    strategy_params={},
    start=datetime(2022, 1, 1),
    end=datetime(2022, 1, 1) + timedelta(days=60),
    step=timedelta(hours=1),
    random_seed=513,
):
    strategy_params = {
        **DEFAULT_STRATEGY_PARAMS,
        **strategy_params,
    }

    parsed_params = StrategyParamsParser.parse_strategy_params(strategy_params)

    engine = SimulationEngine(
        strategy=CoraV1Strategy.from_dict(parsed_params),
        environment=HistoricalCoraEnvironment("ETH"),
        metrics=CoraMetrics(),
        name="metrics_test",
        config=SimulationResultConfig(
            write_step_metrics=False,
            write_end_metrics=False,
            write_log=False,
            write_custom_event_metrics=False,
            meta_log_level="TEST",
        ),
    )

    result = engine.run_simulation(
        start=start,
        end=end,
        step=step,
        random_seed=random_seed,
    )

    return result


def filter_metrics(metric: dict, key: Union[str, Pattern]):
    if isinstance(key, str):
        return {k: v for k, v in metric.items() if key in k}
    else:
        return {k: v for k, v in metric.items() if key.match(k)}


def test_sum_of_loans():
    result = run_simulation()

    for metric in result.step_metrics:
        assert (
            metric["expired_loans_count"]
            == metric["paid_loans_count"] + metric["defaulted_loans_count"]
        )

        assert metric["defaulted_loans_count"] <= metric["expired_loans_count"]


def test_with_low_ltv_no_loans_should_expire_for_this_timeframe():
    result = run_simulation(
        strategy_params={
            "ltv_dist": {
                "type": "dist",
                "name": "uniform",
                "params": {"lower": 0.1, "upper": 0.2},
            },
        }
    )

    min_unpaid_loans = 0

    for metric in result.step_metrics:
        if metric["defaulted_loans_count"] > min_unpaid_loans:
            min_unpaid_loans = metric["defaulted_loans_count"]

    assert min_unpaid_loans == 0


def test_increasing_ltv_will_make_loans_expire():
    result = run_simulation(
        strategy_params={
            "ltv_dist": {
                "type": "dist",
                "name": "uniform",
                "params": {"lower": 0.8, "upper": 0.9},
            },
        }
    )

    min_unpaid_loans = 0

    for metric in result.step_metrics:
        if metric["defaulted_loans_count"] > min_unpaid_loans:
            min_unpaid_loans = metric["defaulted_loans_count"]

    assert min_unpaid_loans > 0


def test_active_loans_sum_histograms():
    result = run_simulation()
    for step in result.step_metrics:
        for distvar in ["ltv", "duration", "start", "size"]:
            metrics = filter_metrics(step, f"hist-active_loans-{distvar}_")
            assert sum(metrics.values()) == step["active_loans_count"]


def test_binned_metrics():
    loan_counter = count(1)
    start_time = datetime(2022, 1, 1)
    loan = Loan(
        start_time=start_time + timedelta(days=10),
        borrowing_fee=0.12,
        net_loan=100.0,
        total_debt=100.0,
        expiration_time=start_time + timedelta(days=20),
        collateral_amount=100,
        loan_id=next(loan_counter),
        borrower_address="0x0",
        paid=False,
        initial_ltv=0.9,
    )

    loans = [loan] * 10

    metrics = CoraMetrics()._get_binned_metrics(
        active_loans=loans,
        loans=loans,
        run_delta=timedelta(days=30),
        run_end=start_time + timedelta(days=30),
    )
    assert metrics["hist-active_loans-duration_0.3_0.4"] == 10
    assert metrics["hist-active_loans-start_0.3_0.4"] == 10
    assert metrics["hist-active_loans-ltv_0.8_0.9"] == 0
    assert metrics["hist-active_loans-ltv_0.9_1.0"] == 10
    assert math.isclose(metrics["dist-loan_fees-duration_0.3_0.4"], 1.2)


def test_random_strategy_params():
    for _ in range(10):
        result = run_simulation(strategy_params=generate_ramdom_strategy_params())

        for step in result.step_metrics:
            loan_duration_hist_active_loans = filter_metrics(
                step, "hist-active_loans-duration"
            )

            loan_duration_hist_unpaid_loans = filter_metrics(
                step, "hist-defaulted_loans-duration"
            )

            ltv_hist_active_loans = filter_metrics(step, "hist-active_loans-ltv")
            ltv_hist_unpaid_loans = filter_metrics(step, "hist-defaulted_loans-ltv")

            assert (
                sum(loan_duration_hist_active_loans.values())
                == step["active_loans_count"]
            )
            assert (
                sum(loan_duration_hist_unpaid_loans.values())
                == step["defaulted_loans_count"]
            )
            assert sum(ltv_hist_active_loans.values()) == step["active_loans_count"]
            assert sum(ltv_hist_unpaid_loans.values()) == step["defaulted_loans_count"]

        assert (
            result.step_metrics[-1]["pool_realized_pnl"]
            == result.end_metrics["pool_pnl"]
        )
