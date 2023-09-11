import sys
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import count
from typing import TYPE_CHECKING, Dict, List, Literal, Type

from protocols.cora.v1.agents import (
    BaseCoraAgent,
    CoraBorrowerAgent,
    CoraLenderAgent,
    CoraPoolManager,
    Wallet,
)
from protocols.cora.v1.business_logic.fee_models import (
    BlackScholesFeeModel,
    KellyFeeModel,
    AaveFeeModel,
    SumBlackScholesAave,
    CombinedBlackScholesAave,
    SumKellyAave,
    CombinedKellyAave,
    CachedKellyFeeModel,
)
from protocols.cora.v1.protocol import CoraV1Protocol
from simulator.strategy import BaseSimulationStrategy
from simulator.utilities.distributions import BaseDistribution

if "pytest" in sys.modules or TYPE_CHECKING:
    from protocols.cora.v1.agents import (
        BaseCoraAgent,
        CoraBorrowerAgent,
        CoraPoolManager,
    )
    from protocols.cora.v1.business_logic.fee_models.base_fee_model import (
        BaseCoraFeeModel,
    )
    from protocols.cora.v1.business_logic.lending_pool import LendingPool
    from protocols.cora.v1.environments import BaseCoraEnvironment


FEE_MODELS: Dict[str, Type["BaseCoraFeeModel"]] = {
    "black_scholes": BlackScholesFeeModel,
    "kelly": KellyFeeModel,
    "traditional": AaveFeeModel,
    "sum_bsm_traditional": SumBlackScholesAave,
    "combined_bsm_traditional": CombinedBlackScholesAave,
    "sum_kelly_traditional": SumKellyAave,
    "combined_kelly_traditional": CombinedKellyAave,
    "cached_kelly": CachedKellyFeeModel,
}


@dataclass(frozen=True)
class CoraV1StategyParameters:
    utilization_parameter: float
    loan_size_dist: BaseDistribution
    loan_start_dist: BaseDistribution
    loan_duration_dist: BaseDistribution
    ltv_dist: BaseDistribution
    max_ltv: float
    max_liquidity: float  # TODO: Make it so can be None for infinite liquidity, default
    initial_lending_amount: float
    fee_model: Literal[
        "black_scholes",
        "kelly",
        "traditional",
        "sum_bsm_traditional",
        "combined_bsm_traditional",
        "cached_kelly",
    ]
    fee_model_update_params: dict
    fee_model_update_interval_seconds: int = 24 * 60 * 60
    genesis_period_seconds: int = 7 * 24 * 60 * 60
    running_period_seconds: int = 30 * 24 * 60 * 60


class CoraV1Strategy(BaseSimulationStrategy):
    LENDING_POOL_NAME = "V1LendingPool"
    UNLIMITED_BALANCE = 1e9

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parameters = CoraV1StategyParameters(**kwargs)
        self.manager_counter = count()
        self.borrower_counter = count()
        self.lender_counter = count()

    def get_initial_protocol(
        self, environment: "BaseCoraEnvironment"
    ) -> "CoraV1Protocol":
        return CoraV1Protocol(environment)

    def get_initial_agents(
        self,
        environment: "BaseCoraEnvironment",
        protocol: "CoraV1Protocol",
    ) -> List["BaseCoraAgent"]:
        # Create a pool manager agent. It will create the lending pool as its only act
        manager_id = f"poolmanager_{next(self.manager_counter):06d}"
        fee_model = FEE_MODELS[self._parameters.fee_model]()
        pool_manager = CoraPoolManager(
            id=manager_id,
            wallet=Wallet(
                address=manager_id,
                primary_balance=0.0,
                secondary_balance=0.0,
            ),
            protocol=protocol,
            environment=environment,
            lending_pool_name=self.LENDING_POOL_NAME,
            fee_model=fee_model,
            fee_model_update_params=self._parameters.fee_model_update_params,
            max_ltv=self._parameters.max_ltv,
            max_liquidity=self._parameters.max_liquidity,
            genesis_period_seconds=self._parameters.genesis_period_seconds,
            running_period_seconds=self._parameters.running_period_seconds,
            parameter_update_seconds=self._parameters.fee_model_update_interval_seconds,
        )

        # Create a lender that will lend capital as soon as there is a lending pool
        lender_id = f"lender_{next(self.lender_counter):06d}"
        lender = CoraLenderAgent(
            id=lender_id,
            protocol=protocol,
            environment=environment,
            amount=self._parameters.initial_lending_amount,
            wallet=Wallet(
                address=lender_id,
                primary_balance=self.UNLIMITED_BALANCE,
                secondary_balance=self.UNLIMITED_BALANCE,
            ),
        )

        return [pool_manager, lender]

    def update_agents(
        self,
        agents: List["BaseCoraAgent"],
        protocol: "CoraV1Protocol",
        environment: "BaseCoraEnvironment",
        time_step: timedelta,
    ) -> List["BaseCoraAgent"]:
        for lending_pool in protocol.get_lending_pools():
            # Determine if the lending pool has entered a new running period
            if lending_pool.is_new_cycle():
                # Delete all borrower agents from this lending pool
                for agent in agents:
                    if isinstance(agent, CoraBorrowerAgent):
                        agents.remove(agent)
                # Create new borrower agents
                new_agents = self._create_borrower_agents(
                    protocol,
                    environment,
                    lending_pool,
                    time_step,
                )
                agents.extend(new_agents)
        return agents

    def _create_borrower_agents(
        self,
        protocol: "CoraV1Protocol",
        environment: "BaseCoraEnvironment",
        lending_pool: "LendingPool",
        time_step: timedelta,
    ) -> List["CoraBorrowerAgent"]:

        available_liquidity = lending_pool._available_amount
        target_total_loans = (
            self._parameters.utilization_parameter * available_liquidity
        )
        running_period = lending_pool._running_period

        new_agents = []
        total_loan_size = 0
        while True:
            borrower_loan_size = self._parameters.loan_size_dist.sample()
            if total_loan_size + borrower_loan_size > target_total_loans:
                break  # Next borrower would go over target total loans, so stop

            borrower_id = f"borrower_{next(self.borrower_counter):06d}"

            loan_start_factor = self._parameters.loan_start_dist.sample()
            loan_start_delta = loan_start_factor * running_period
            loan_start = environment.get_time() + loan_start_delta

            max_duration = running_period - loan_start_delta
            loan_duration_factor = self._parameters.loan_duration_dist.sample()
            loan_duration = max(
                (loan_duration_factor * (max_duration - 2 * time_step) + time_step),
                time_step,
            )

            ltv_factor = self._parameters.ltv_dist.sample()
            ltv = min(ltv_factor, self._parameters.max_ltv - 1e-9)

            borrower = CoraBorrowerAgent(
                id=borrower_id,
                protocol=protocol,
                environment=environment,
                wallet=Wallet(
                    address=borrower_id,
                    primary_balance=self.UNLIMITED_BALANCE,
                    secondary_balance=self.UNLIMITED_BALANCE,
                ),
                lending_pool_name=lending_pool.name,
                loan_size=borrower_loan_size,
                loan_start=loan_start,
                loan_duration=loan_duration,
                ltv=ltv,
                repay_margin=time_step,
            )

            total_loan_size += borrower_loan_size
            new_agents.append(borrower)

        return new_agents


@dataclass(frozen=True)
class CoraV2StategyParameters:
    borrower_demand_ratio: float
    loan_size_dist: BaseDistribution
    loan_start_dist: BaseDistribution
    loan_duration_dist: BaseDistribution
    ltv_dist: BaseDistribution
    max_ltv: float
    max_liquidity: float
    initial_lending_amount: float
    fee_model: Literal[
        "black_scholes",
        "kelly",
        "traditional",
        "sum_bsm_traditional",
        "combined_bsm_traditional",
        "cached_kelly",
    ]
    fee_model_update_params: dict
    fee_model_update_interval_seconds: int = 24 * 60 * 60
    genesis_period_seconds: int = 7 * 24 * 60 * 60
    running_period_seconds: int = 30 * 24 * 60 * 60


class CoraV2Strategy(BaseSimulationStrategy):
    LENDING_POOL_NAME = "V2LendingPool"
    UNLIMITED_BALANCE = 1e9

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parameters = CoraV2StategyParameters(**kwargs)
        self.manager_counter = count()
        self.borrower_counter = count()
        self.lender_counter = count()

    def get_initial_protocol(
        self, environment: "BaseCoraEnvironment"
    ) -> "CoraV1Protocol":
        return CoraV1Protocol(environment)

    def get_initial_agents(
        self,
        environment: "BaseCoraEnvironment",
        protocol: "CoraV1Protocol",
    ) -> List["BaseCoraAgent"]:
        # Create a pool manager agent. It will create the lending pool as its only act
        manager_id = f"poolmanager_{next(self.manager_counter):06d}"
        fee_model = FEE_MODELS[self._parameters.fee_model]()
        pool_manager = CoraPoolManager(
            id=manager_id,
            wallet=Wallet(
                address=manager_id,
                primary_balance=0.0,
                secondary_balance=0.0,
            ),
            protocol=protocol,
            environment=environment,
            lending_pool_name=self.LENDING_POOL_NAME,
            fee_model=fee_model,
            fee_model_update_params=self._parameters.fee_model_update_params,
            max_ltv=self._parameters.max_ltv,
            max_liquidity=self._parameters.max_liquidity,
            genesis_period_seconds=self._parameters.genesis_period_seconds,
            running_period_seconds=self._parameters.running_period_seconds,
            parameter_update_seconds=self._parameters.fee_model_update_interval_seconds,
        )

        # Create a lender that will lend capital as soon as there is a lending pool
        lender_id = f"lender_{next(self.lender_counter):06d}"
        lender = CoraLenderAgent(
            id=lender_id,
            protocol=protocol,
            environment=environment,
            amount=self._parameters.initial_lending_amount,
            wallet=Wallet(
                address=lender_id,
                primary_balance=self.UNLIMITED_BALANCE,
                secondary_balance=self.UNLIMITED_BALANCE,
            ),
        )

        return [pool_manager, lender]

    def update_agents(
        self,
        agents: List["BaseCoraAgent"],
        protocol: "CoraV1Protocol",
        environment: "BaseCoraEnvironment",
        time_step: timedelta,
    ) -> List["BaseCoraAgent"]:
        for lending_pool in protocol.get_lending_pools():
            # Determine if the lending pool has entered a new running period
            if lending_pool.is_new_cycle():
                # Delete all borrower agents from this lending pool
                for agent in agents:
                    if isinstance(agent, CoraBorrowerAgent):
                        agents.remove(agent)
                # Create new borrower agents
                new_agents = self._create_borrower_agents(
                    protocol,
                    environment,
                    lending_pool,
                    time_step,
                )
                agents.extend(new_agents)
        return agents

    def _create_borrower_agents(
        self,
        protocol: "CoraV1Protocol",
        environment: "BaseCoraEnvironment",
        lending_pool: "LendingPool",
        time_step: timedelta,
    ) -> List["CoraBorrowerAgent"]:

        available_liquidity = lending_pool._available_amount
        running_period = lending_pool._running_period

        new_agents = []
        marginal_utilization_sum = 0.0
        while True:
            borrower_loan_size = self._parameters.loan_size_dist.sample()

            loan_start_factor = self._parameters.loan_start_dist.sample()
            loan_start_delta = loan_start_factor * running_period
            loan_start = environment.get_time() + loan_start_delta

            max_duration = running_period - loan_start_delta
            loan_duration_factor = self._parameters.loan_duration_dist.sample()
            loan_duration = max(
                (loan_duration_factor * (max_duration - 2 * time_step) + time_step),
                time_step,
            )

            liquidity_ratio = borrower_loan_size / available_liquidity
            duration_ratio = loan_duration / running_period

            marginal_utilization = liquidity_ratio * duration_ratio
            if (
                marginal_utilization_sum + marginal_utilization
                > self._parameters.borrower_demand_ratio
            ):
                break

            borrower_id = f"borrower_{next(self.borrower_counter):06d}"
            ltv_factor = self._parameters.ltv_dist.sample()
            ltv = min(ltv_factor, self._parameters.max_ltv - 1e-9)

            borrower = CoraBorrowerAgent(
                id=borrower_id,
                protocol=protocol,
                environment=environment,
                wallet=Wallet(
                    address=borrower_id,
                    primary_balance=self.UNLIMITED_BALANCE,
                    secondary_balance=self.UNLIMITED_BALANCE,
                ),
                lending_pool_name=lending_pool.name,
                loan_size=borrower_loan_size,
                loan_start=loan_start,
                loan_duration=loan_duration,
                ltv=ltv,
                repay_margin=time_step,
            )

            marginal_utilization_sum += marginal_utilization
            new_agents.append(borrower)

        return new_agents
