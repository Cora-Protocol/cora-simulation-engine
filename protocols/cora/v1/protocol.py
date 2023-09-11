from datetime import timedelta
from typing import List

from protocols.cora.v1.business_logic.fee_models.base_fee_model import BaseCoraFeeModel
from protocols.cora.v1.business_logic.lending_pool import LendingPool
from protocols.cora.v1.environments import BaseCoraEnvironment
from simulator.protocol import BaseProtocol


class PriceFeed:
    def __init__(self):
        pass


class CoraV1Protocol(BaseProtocol):
    def __init__(self, environment: BaseCoraEnvironment) -> None:
        self._total_lending_pools = 0
        self._lending_pools: dict[str, LendingPool] = {}
        self._environment = environment

    def take_step(self, time_step: timedelta) -> List[dict]:
        # Take the step in each lending pool
        events = []
        for lending_pool in self.get_lending_pools():
            events.extend(lending_pool.take_step(time_step))
        return events

    def create_lending_pool(
        self,
        name: str,
        fee_model: BaseCoraFeeModel,
        max_ltv: float,
        max_liquidity: float,
        genesis_period_seconds: int,
        running_period_seconds: int,
        min_liquidity: float = 0.0,
        min_loan_amount: float = 0.0,
        min_loan_period_seconds: int = 0,
        min_position_size: float = 0.0,
    ):

        assert name not in self._lending_pools, "Lending pool already exists"
        assert min_loan_amount >= 0, "Minimum loan amount must be non-negative"
        assert min_loan_period_seconds >= 0, "Minimum loan period must be non-negative"
        assert min_position_size >= 0, "Minimum position size must be non-negative"
        assert genesis_period_seconds >= 0, "Genesis period must be non-negative"
        assert running_period_seconds >= 0, "Running period must be non-negative"
        assert min_liquidity >= 0, "Minimum liquidity must be non-negative"
        assert max_ltv > 0, "Maximum LTV must be positive"
        assert max_ltv <= 1, "Maximum LTV must be less than or equal to 1"
        assert max_liquidity > 0, "Maximum liquidity must be positive"

        lending_pool = LendingPool(
            name=name,
            environment=self._environment,
            fee_model=fee_model,
            max_ltv=max_ltv,
            max_liquidity=max_liquidity,
            genesis_period_seconds=genesis_period_seconds,
            running_period_seconds=running_period_seconds,
            min_liquidity=min_liquidity,
            min_loan_amount=min_loan_amount,
            min_loan_period_seconds=min_loan_period_seconds,
            min_position_size=min_position_size,
        )

        self._lending_pools[name] = lending_pool
        self._total_lending_pools += 1

    def get_lending_pools(self) -> List[LendingPool]:
        return list(self._lending_pools.values())

    def get_lending_pool(self, name: str) -> LendingPool:
        return self._lending_pools[name]
