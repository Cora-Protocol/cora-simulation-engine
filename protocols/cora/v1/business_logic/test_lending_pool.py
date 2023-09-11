from datetime import timedelta, datetime
from unittest import TestCase

import pytest
from protocols.cora.v1.agents import CoraLenderAgent, Wallet
from protocols.cora.v1.business_logic.fee_models.base_fee_model import BaseCoraFeeModel
from protocols.cora.v1.business_logic.lending_pool import LendingPool, LendingPoolStatus
from protocols.cora.v1.environments import HistoricalCoraEnvironment
from protocols.cora.v1.exceptions.exceptions import LendingPoolNotRunning
from protocols.cora.v1.protocol import CoraV1Protocol


class TestLendingPool(TestCase):
    def setUp(self):
        environment = HistoricalCoraEnvironment("ETH")
        environment.set_time(datetime.fromisoformat("2022-01-01T00:00:00"))
        environment.load_data_until(datetime.fromisoformat("2022-01-02T00:00:00"))
        self.lending_pool = LendingPool(
            name="LendingPool",
            environment=environment,
            fee_model=BaseCoraFeeModel(),
            max_ltv=100,
            max_liquidity=1000,
            genesis_period_seconds=0,
            running_period_seconds=180,
        )

        self.lender = CoraLenderAgent(
            id="lender_01",
            protocol=CoraV1Protocol(environment),
            environment=environment,
            wallet=Wallet("mock_address", 1000.0, 2000.0),
            amount=500.0,
        )

    def test_pool_gets_correctly_instanciated(self):
        assert self.lending_pool._status == LendingPoolStatus.GENESIS

    def test_running_only_decorator(self):
        with pytest.raises(LendingPoolNotRunning):
            self.lending_pool.signal_withdrawal(self.lender, 1)

    def test_pool_can_be_started(self):
        self.lending_pool._environment.take_step(timedelta(seconds=60))
        self.lending_pool.take_step(timedelta(seconds=60))
        assert self.lending_pool._status == LendingPoolStatus.RUNNING
