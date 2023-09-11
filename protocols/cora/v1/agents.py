import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List

from simulator.agent import BaseAgent
from simulator.models.action_info import ActionInfo

if ("pytest" in sys.modules) and TYPE_CHECKING:
    from protocols.cora.v1.business_logic.fee_models.base_fee_model import (
        BaseCoraFeeModel,
    )
    from protocols.cora.v1.environments import BaseCoraEnvironment
    from protocols.cora.v1.protocol import CoraV1Protocol


# TODO: This is probably more useful as a common utility, move where appropriate
@dataclass
class Wallet:
    address: str
    primary_balance: float = 0.0
    secondary_balance: float = 0.0


class BaseCoraAgent(BaseAgent):
    def __init__(
        self,
        id: str,
        protocol: "CoraV1Protocol",
        environment: "BaseCoraEnvironment",
        wallet: Wallet,
    ):
        self.id = id
        self.protocol = protocol
        self.environment = environment
        self.wallet = wallet


class CoraBorrowerAgent(BaseCoraAgent):
    _priority = 2

    def __init__(
        self,
        id: int,
        protocol: "CoraV1Protocol",
        environment: "BaseCoraEnvironment",
        wallet: Wallet,
        lending_pool_name: str,
        loan_size: float,
        loan_start: datetime,
        loan_duration: timedelta,
        ltv: float,
        repay_margin: timedelta,
    ):
        super().__init__(id, protocol, environment, wallet)
        self.lending_pool_name = lending_pool_name
        self.loan_size = loan_size
        self.loan_start = loan_start
        self.loan_duration = loan_duration
        self.ltv = ltv
        self.repay_margin = repay_margin
        self._has_borrowed = False
        self._has_expired = False

    def act(self) -> List[ActionInfo]:
        current_time = self.environment.get_time()
        actions = []
        # Ask for a loan
        if (current_time >= self.loan_start) and not self._has_borrowed:
            lending_pool = self.protocol.get_lending_pool(self.lending_pool_name)
            if (lending_pool.get_current_available_amount() > self.loan_size) and (
                self.loan_duration <= (lending_pool._next_cycle_time - current_time)
            ):
                spot_price = self.environment.get_price()
                collateral_amount = self.loan_size / self.ltv / spot_price
                self.loan = lending_pool.borrow(
                    borrower=self,
                    borrow_amount=self.loan_size,
                    collateral_amount=collateral_amount,
                    loan_period=self.loan_duration,
                )
                message = (
                    f"{current_time}: Agent {self.id} borrowed {self.loan_size:.2f} "
                    f"with an LTV of {self.ltv * 100:.2f}% and a loan duration of "
                    f"{self.loan_duration.total_seconds()/86400:.2f} days on LP "
                    f"{lending_pool.name}, creating loan {self.loan.loan_id}"
                )
                action = ActionInfo(
                    message=message,
                    agent_id=self.id,
                    time=current_time,
                    type="borrow",
                    extra={
                        "loan_duration": self.loan_duration.days,
                        "ltv": self.ltv,
                        "loan_size": self.loan_size,
                        "net_loan": self.loan.net_loan,
                        "collateral_amount": collateral_amount,
                        "loan_fee": self.loan.borrowing_fee,
                        "total_debt": self.loan.total_debt,
                        "lending_pool": lending_pool.name,
                        "loan_id": self.loan.loan_id,
                    },
                )
                actions.append(action)
                self._has_borrowed = True

        # Return the loan if beneficial
        if (
            self._has_borrowed
            and (
                current_time >= self.loan_start + self.loan_duration - self.repay_margin
            )
            and not self._has_expired
        ):
            # Decide whether to return the loan
            spot_price = self.environment.get_price()
            collateral_value = self.loan.collateral_amount * spot_price
            if collateral_value > self.loan.total_debt:
                lending_pool = self.protocol.get_lending_pool(self.lending_pool_name)
                lending_pool.repay(self, self.loan.loan_id)
                message = (
                    f"{current_time}: Agent {self.id} returned "
                    f"{self.loan.total_debt:.2f} for loan {self.loan.loan_id}",
                )
                action = ActionInfo(
                    message=message,
                    agent_id=self.id,
                    time=current_time,
                    type="repay",
                    extra={
                        "collateral_value": collateral_value,
                        "loan_debt": self.loan.total_debt,
                        "loan_fee": self.loan.borrowing_fee,
                        "lending_pool": lending_pool.name,
                        "loan_id": self.loan.loan_id,
                    },
                )

                actions.append(action)
            else:
                message = (
                    f"{current_time}: Agent {self.id} let {self.loan.loan_id} expire "
                    f"because his collateral was {collateral_value} vs "
                    f"{self.loan.total_debt}"
                )
                action = ActionInfo(
                    message=message,
                    agent_id=self.id,
                    time=current_time,
                    type="let_expire",
                    extra={
                        "collateral_value": collateral_value,
                        "loan_debt": self.loan.total_debt,
                        "loan_fee": self.loan.borrowing_fee,
                        "lending_pool": self.lending_pool_name,
                        "loan_id": self.loan.loan_id,
                    },
                )
                actions.append(action)
            self._has_expired = True
        return actions


class CoraLenderAgent(BaseCoraAgent):
    _priority = 1

    def __init__(
        self,
        id: str,
        protocol: "CoraV1Protocol",
        environment: "BaseCoraEnvironment",
        wallet: Wallet,
        amount: float,
    ):
        super().__init__(id, protocol, environment, wallet)
        self.amount = amount
        self._has_lended = False

    def act(self) -> List[ActionInfo]:
        actions = []
        if not self._has_lended:
            current_time = self.environment.get_time()
            for lending_pool in self.protocol.get_lending_pools():
                lending_pool.deposit(
                    lender=self,
                    amount=self.amount,
                )
                message = (
                    f"{current_time}: Agent {self.id} has deposited {self.amount} on"
                    f" Lending Pool {lending_pool.name}"
                )
                action = ActionInfo(
                    message=message,
                    agent_id=self.id,
                    time=current_time,
                    type="deposit",
                    extra={
                        "amount": self.amount,
                        "lending_pool": lending_pool.name,
                    },
                )
                actions.append(action)
                self._has_lended = True
                break
        return actions


class CoraPoolManager(BaseCoraAgent):
    _priority = 0

    def __init__(
        self,
        id: str,
        protocol: "CoraV1Protocol",
        environment: "BaseCoraEnvironment",
        wallet: Wallet,
        lending_pool_name: str,
        fee_model: "BaseCoraFeeModel",
        fee_model_update_params: dict,
        max_ltv: float,
        max_liquidity: float,
        genesis_period_seconds: int,
        running_period_seconds: int,
        parameter_update_seconds: int,
    ):
        super().__init__(id, protocol, environment, wallet)

        assert max_ltv < 1.0, "max_ltv must be less than 1.0"
        assert max_liquidity > 0.0, "max_liquidity must be greater than 0.0"

        self.name = lending_pool_name
        self.fee_model = fee_model
        self.fee_model_update_params = fee_model_update_params
        self.max_ltv = max_ltv
        self.max_liquidity = max_liquidity
        self.genesis_period_seconds = genesis_period_seconds
        self.running_period_seconds = running_period_seconds
        self.parameter_update_period = timedelta(seconds=parameter_update_seconds)
        self.next_parameter_update = self.environment.get_time()

    def act(self) -> List[ActionInfo]:
        # If there is no lending pool in the protocol, create it
        actions = []
        if not self.protocol.get_lending_pools():
            current_time = self.environment.get_time()
            self.protocol.create_lending_pool(
                name=self.name,
                fee_model=self.fee_model,
                max_ltv=self.max_ltv,
                max_liquidity=self.max_liquidity,
                genesis_period_seconds=self.genesis_period_seconds,
                running_period_seconds=self.running_period_seconds,
            )

            fee_model_name = self.fee_model.__class__.__name__
            message = (
                f"{current_time}: Agent {self.id} has createdlending pool {self.name} "
                f"with {fee_model_name}, a max_ltv of {self.max_ltv*100:.2f}%, a "
                f"max_liquitidy of {self.max_liquidity:.2f}, a genesis period of "
                f"{self.genesis_period_seconds/86400:.2f} days, and a running period "
                f"of {self.running_period_seconds/86400:.2f} days"
            )
            action = ActionInfo(
                message=message,
                agent_id=self.id,
                time=current_time,
                type="create_lending_pool",
                extra={
                    "fee_model": fee_model_name,
                    "max_ltv": self.max_ltv,
                    "max_liquidity": self.max_liquidity,
                    "genesis_period_seconds": self.genesis_period_seconds,
                    "running_period_seconds": self.running_period_seconds,
                    "lending_pool": self.name,
                },
            )
            actions.append(action)

        if self.environment.get_time() >= self.next_parameter_update:
            self.next_parameter_update += self.parameter_update_period
            lending_pool = self.protocol.get_lending_pool(self.name)
            fee_model_params = self.fee_model.get_parameters(
                self.environment, **self.fee_model_update_params
            )
            lending_pool._fee_model.update_parameters(**fee_model_params)

            current_time = self.environment.get_time()
            message = (
                f"{current_time}: Agent {self.id} has updated the fee parameters of "
                f"Lending Pool {self.name}"
            )
            action = ActionInfo(
                message=message,
                agent_id=self.id,
                time=current_time,
                type="update_fee_parameters",
                extra={
                    "lending_pool": self.name,
                    "fee_model": self.fee_model.__class__.__name__,
                    "fee_model_params": fee_model_params,
                },
            )
            actions.append(action)
        return actions
