import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Dict, List

from protocols.cora.v1.exceptions.exceptions import (
    InsufficientBalanceError,
    InsufficientLiquidityError,
    InsuficientCollateralError,
    InvalidLoanId,
    InvalidLoanPeriodLong,
    InvalidLoanPeriodShort,
    LendingPoolNotRunning,
    LoanAmountTooLow,
    LoanExpiredError,
    NonExistingBorrowerAddress,
)
from simulator.metrics.calculations import safe_divide
from simulator.models.event_info import EventInfo

if ("pytest" in sys.modules) or TYPE_CHECKING:
    from protocols.cora.v1.agents import CoraBorrowerAgent, CoraLenderAgent
    from protocols.cora.v1.business_logic.fee_models.base_fee_model import (
        BaseCoraFeeModel,
    )
    from protocols.cora.v1.environments import BaseCoraEnvironment


def running_only(func):
    @wraps(func)
    def wrapper(self: "LendingPool", *args, **kwargs):
        if self.get_status() != LendingPoolStatus.RUNNING:
            raise LendingPoolNotRunning
        return func(self, *args, **kwargs)

    return wrapper


class LendingPoolStatus(Enum):
    """
    Status of the lending pool.
    """

    GENESIS = "genesis"
    RUNNING = "running"


@dataclass
class Loan:
    start_time: datetime
    borrowing_fee: float
    net_loan: float
    total_debt: float
    expiration_time: datetime
    collateral_amount: float
    loan_id: str
    borrower_address: str
    initial_ltv: float
    paid: bool = False

    def is_expired(self, environment: "BaseCoraEnvironment") -> bool:
        return self.expiration_time < environment.get_time()

    def get_duration(self) -> timedelta:
        return self.expiration_time - self.start_time

    def get_sizedays(self) -> float:
        days = self.get_duration().total_seconds() / 86400
        return days * self.net_loan


@dataclass
class CycleData:
    initial_liquidity: float
    remaining_liquidity: float
    total_reclaimed_collateral: float
    total_fees_earned: float
    final_collateral_price: float
    final_collateral_value: float
    average_utilization: float
    normalized_utilization: float
    loans: List[Loan]


class LendingPool:
    def __init__(
        self,
        name: str,
        environment: "BaseCoraEnvironment",
        fee_model: "BaseCoraFeeModel",
        max_ltv: float,
        max_liquidity: float,
        genesis_period_seconds: int,
        running_period_seconds: int,
        min_liquidity: float = 0.0,
        min_loan_amount: float = 0.0,
        min_loan_period_seconds: int = 0,
        min_position_size: float = 0.0,
    ):
        self.name = name
        self._environment = environment
        self._fee_model = fee_model
        self._max_ltv = max_ltv
        self._max_liquidity = max_liquidity
        self._genesis_period = timedelta(seconds=genesis_period_seconds)
        self._running_period = timedelta(seconds=running_period_seconds)

        self._min_liquidity = min_liquidity
        self._min_loan_amount = min_loan_amount
        self._min_load_period = timedelta(seconds=min_loan_period_seconds)
        self._min_position_size = min_position_size

        self._status = LendingPoolStatus.GENESIS  # Initial status is always genesis
        self._next_cycle_time = environment.get_time() + self._genesis_period

        # Lender data
        self._pending_deposits: Dict[str, float] = {}  # address:amount
        self._signaled_withdrawals: Dict[str, float] = {}  # address:ratio
        self._pending_withdrawals: Dict[str, float] = {}  # address:amount
        self._reclaimed_collateral: Dict[str, float] = {}  # address:amount
        self._deposits: Dict[str, float] = {}  # address:amount

        self._borrower_loans: Dict[str, List[str]] = {}  # address:[loan_ids]
        self._loans: Dict[str, Loan] = {}  # loan_id:Loan
        self._utilizations: List[float] = []  # utilizations of the last cycle
        self._cycle_history: Dict[int, CycleData] = {}  # cycle_number:CycleData

        self._total_deposits: float = 0.0
        self._total_collateral_locked: float = 0.0
        self._available_amount: float = 0.0
        self._borrowed_amount: float = 0.0
        self._total_fees_earned: float = 0.0

        self._is_new_cycle = False
        self._cycle_count = 0

    def take_step(self, time_step: timedelta) -> List[EventInfo]:
        time = self._environment.get_time()
        events: List[EventInfo] = []

        self._utilizations.append(self.get_utilization())

        if time >= self._next_cycle_time:  # Cycle has changed
            self._is_new_cycle = True
            if self._status == LendingPoolStatus.GENESIS:  # First cycle
                self._status = LendingPoolStatus.RUNNING
                self._cycle_count += 1
                # Deposit the pending deposits
                self._deposits = self._pending_deposits.copy()
                message = (
                    f"{time}: Lending pool {self.name} ended its genesis period and "
                    f"started running, with {self._total_deposits} in deposits."
                )
                event_info = EventInfo(
                    message=message,
                    time=time,
                    type="lending_pool_genesis_period_ended",
                    extra={
                        "total_deposits": self._total_deposits,
                        "cycle_count": self._cycle_count,
                        "lending_pool": self.name,
                    },
                )
                events.append(event_info)

            else:  # Current cycle is over, execute settlements
                self._cycle_count += 1

                # Redistribute reclaimed collateral and liquidity proportionally to the cycle's deposits
                total_reclaimed_collateral = self._total_collateral_locked
                remaining_liquidity = self._available_amount
                lenders_final_liquidity: Dict[str, float] = {}
                for address, deposit_amount in self._deposits.items():
                    ownership_ratio = deposit_amount / self._total_deposits

                    # Reclaim liquidity and assign in proportion to the ownership ratio
                    lender_final_liquidity = ownership_ratio * remaining_liquidity
                    lenders_final_liquidity[address] = lender_final_liquidity

                    # Reclaim collateral and assign in proportion to the ownership ratio
                    reclaimed_collateral = ownership_ratio * total_reclaimed_collateral
                    if address not in self._reclaimed_collateral:
                        self._reclaimed_collateral[address] = 0
                    self._reclaimed_collateral[address] += reclaimed_collateral

                # Execute previously signaled liquidity withdrawals
                for address, withdrawal_ratio in self._signaled_withdrawals.items():
                    lender_liquidity = lenders_final_liquidity[address]
                    lender_withdrawal = lender_liquidity * withdrawal_ratio

                    # Add to pending withdrawals
                    lenders_final_liquidity[address] -= lender_withdrawal
                    if address not in self._pending_withdrawals:
                        self._pending_withdrawals[address] = 0.0
                    self._pending_withdrawals[address] += lender_withdrawal

                # Add pending deposits to deposits
                self._deposits = self._pending_deposits.copy()

                # Add liquidity not withdrawn from previous cycle to deposits
                for address, amount in lenders_final_liquidity.items():
                    if amount > 0:
                        if address not in self._deposits:
                            self._deposits[address] = 0.0
                        self._deposits[address] += amount

                average_utilization = sum(self._utilizations) / len(self._utilizations)
                pool_sizedays = (
                    self._total_deposits * self._running_period.total_seconds() / 86400
                )
                normalized_utilization = (
                    sum(loan.get_sizedays() for loan in self._loans.values())
                    / pool_sizedays
                )
                # Record cycle history
                self._cycle_history[self._cycle_count - 1] = CycleData(
                    initial_liquidity=self._total_deposits,
                    remaining_liquidity=self._available_amount,
                    total_reclaimed_collateral=total_reclaimed_collateral,
                    total_fees_earned=self._total_fees_earned,
                    final_collateral_price=self._environment.get_price(),
                    final_collateral_value=self._environment.get_price()
                    * total_reclaimed_collateral,
                    average_utilization=average_utilization,
                    normalized_utilization=normalized_utilization,
                    loans=list(self._loans.values()),
                )

                message = (
                    f"{time}: Lending pool {self.name} ended its running period "
                    f"#{self._cycle_count - 1} and started #{self._cycle_count}."
                )
                event = EventInfo(
                    message=message,
                    time=time,
                    type="lending_pool_running_period_ended",
                    extra={
                        "total_deposits": sum(self._deposits.values()),
                        "cycle_number": self._cycle_count - 1,
                        "lending_pool": self.name,
                    },
                )
                events.append(event)

            # Reset lending pool data
            self._pending_deposits = {}
            self._signaled_withdrawals = {}

            self._borrower_loans = {}
            self._loans = {}

            self._total_deposits = sum(self._deposits.values())
            self._total_collateral_locked = 0.0
            self._available_amount = self._total_deposits
            self._borrowed_amount = 0.0
            self._total_fees_earned = 0.0

            self._utilizations = []

            self._next_cycle_time += self._running_period

        else:
            self._is_new_cycle = False

        return events

    def get_utilization(self) -> float:
        return safe_divide(
            self._borrowed_amount, self._borrowed_amount + self._available_amount
        )

    def get_status(self) -> LendingPoolStatus:
        return self._status

    def get_all_loans(self) -> List[Loan]:
        return list(self._loans.values())

    def is_new_cycle(self) -> bool:
        return self._is_new_cycle

    def get_borrower_loan_ids(self, borrower: "CoraBorrowerAgent") -> List[str]:
        return self._borrower_loans.get(borrower.wallet.address, [])

    def get_loan_by_id(self, loan_id: str) -> Loan:
        return self._loans.get(loan_id, None)

    def get_current_utilization(self):  # TODO: Check definition with @Edson
        if self._available_amount == 0:
            return 1.0
        total_amount = self._available_amount + self._borrowed_amount
        utilization = self._borrowed_amount / total_amount
        return utilization if utilization > 0 else 0.0

    def update_collateral_price(self, price: float):
        self._collateral_price = price

    def update_fee_model(self, model: "BaseCoraFeeModel"):
        self._fee_model = model

    def get_current_available_amount(self) -> float:
        return self._available_amount

    def calculate_fee(
        self, borrow_amount: float, collateral_value: float, loan_period: timedelta
    ) -> float:
        ltv = borrow_amount / collateral_value
        utilization = self.get_current_utilization()
        fee_percentage = self._fee_model.get_fee(ltv, utilization, loan_period)
        return fee_percentage * borrow_amount

    def calculate_loan_id(
        self, borrower: "CoraBorrowerAgent", expiration_time: int
    ) -> str:
        # NOTE: Actually, hash of protocol address, borrower address, and expiration time
        return f"{self.name}-{borrower.wallet.address}-{expiration_time}"

    def deposit(self, lender: "CoraLenderAgent", amount: float):
        # Validate that the amount is above the minimum position
        # TODO: Implement this check once we get a definition from Cora, not important

        # Valudate that the lender has enough balance to deposit this
        if lender.wallet.primary_balance < amount:
            raise InsufficientBalanceError()

        # Add the lender to the list of pending deposits
        if not lender.wallet.address in self._pending_deposits:
            self._pending_deposits[lender.wallet.address] = 0

        # Lock the balance, to be added starting in the next cycle
        lender.wallet.primary_balance -= amount
        self._pending_deposits[lender.wallet.address] += amount

    @running_only
    def signal_withdrawal(self, lender: "CoraLenderAgent", ratio: float):
        # Validate that the ratio is between 0 and 1
        assert 0 <= ratio <= 1, "Withdrawal ratio must be between 0 and 1"

        # Validate that the lender has deposits to withdraw
        assert (
            lender.wallet.address in self._deposits
        ), "Lender has no deposits to withdraw"

        # Add lender to pending withdrawals, or overwrite the previous value
        self._signaled_withdrawals[lender.wallet.address] = ratio

    def withdraw_liquidity(self, lender: "CoraLenderAgent", amount: float):
        assert amount > 0, "Withdrawal amount must be greater than 0"

        # Amount is withdrawn from pending deposits, and them from pending withdrawals
        pending_deposits = self._pending_deposits.get(lender.wallet.address, 0)
        pending_withdrawals = self._pending_withdrawals.get(lender.wallet.address, 0)
        total_available = pending_deposits + pending_withdrawals

        # Validate that the lender has enough balance to withdraw this
        assert total_available >= amount

        # Remove quantity from pending deposits
        if amount < pending_deposits:
            self._pending_deposits[lender.wallet.address] -= amount
        else:
            del self._pending_deposits[lender.wallet.address]
            remaining_amount = amount - pending_deposits
            # Remove remaining quantity from pending withdrawals
            self._pending_withdrawals[lender.wallet.address] -= remaining_amount

        # Add the amount to the lender's balance
        lender.wallet.primary_balance += amount

    def withdraw_collateral(self, lender: "CoraLenderAgent", amount: float):
        assert amount > 0, "Withdrawal amount must be greater than 0"

        reclaimed_collateral = self._reclaimed_collateral.get(lender.wallet.address, 0)

        # Validate that the lender has enough balance to withdraw this
        assert reclaimed_collateral >= amount, "Insufficient reclaimed collateral"

        # Remove quantity from reclaimed collateral
        self._reclaimed_collateral[lender.wallet.address] -= amount

        # Add the amount to the lender's balance
        lender.wallet.secondary_balance += amount

    @running_only
    def borrow(
        self,
        borrower: "CoraBorrowerAgent",
        borrow_amount: float,
        collateral_amount: float,
        loan_period: timedelta,
    ) -> Loan:

        price = self._environment.get_price()
        time = self._environment.get_time()

        total_collateral_value = collateral_amount * price
        ltv = borrow_amount / total_collateral_value
        max_amount_available_to_borrow = total_collateral_value * self._max_ltv

        # Validate that the amount requested is higher than the minimum allowed
        if borrow_amount < self._min_loan_amount:
            raise LoanAmountTooLow()

        # Validate that the LTV of the requested loan is within the range of the pool
        if borrow_amount > max_amount_available_to_borrow:
            raise InsuficientCollateralError(borrow_amount, ltv)

        # Validate that the borrowe has enough collateral
        if borrower.wallet.secondary_balance < collateral_amount:
            raise InsufficientBalanceError()

        # Validate that lending pool has enough liquidity
        if self._available_amount < borrow_amount:
            raise InsufficientLiquidityError()

        # Validate that the period is longer than the minimum period
        if loan_period < self._min_load_period:
            print(
                f"Loan period {loan_period} is shorter than the minimum period {self._min_load_period}"
            )
            raise InvalidLoanPeriodShort()

        # Validate that the period is within the remaining running time
        if loan_period > (self._next_cycle_time - time):
            print(
                f"Loan period is longer than remaining running time: {loan_period} > {self._next_cycle_time}"
            )
            raise InvalidLoanPeriodLong()

        # Calculate the borrowing fee
        borrowing_fee = self.calculate_fee(
            borrow_amount, total_collateral_value, loan_period
        )

        # Update properties of the lending pool
        self._total_collateral_locked += collateral_amount
        # NOTE: Not strictly correct, but how Cora implements it ¯\_(ツ)_/¯
        net_loan_amount = borrow_amount - borrowing_fee
        self._available_amount -= net_loan_amount
        self._borrowed_amount += net_loan_amount

        # Update properties of borrower
        borrower.wallet.secondary_balance -= collateral_amount
        borrower.wallet.primary_balance += net_loan_amount

        # Store loan information

        expiration_time = time + loan_period
        expiration_timestamp = int(expiration_time.timestamp())
        loan_id = self.calculate_loan_id(borrower, expiration_timestamp)

        # Store the which loans this borrower has active
        if not borrower.wallet.address in self._borrower_loans:
            self._borrower_loans[borrower.wallet.address] = []
        self._borrower_loans[borrower.wallet.address].append(loan_id)

        initial_loan_ltv = borrow_amount / (collateral_amount * price)
        # Store the loan information
        loan = Loan(
            start_time=self._environment.get_time(),
            borrowing_fee=borrowing_fee,
            net_loan=net_loan_amount,
            total_debt=borrow_amount,
            expiration_time=expiration_time,
            collateral_amount=collateral_amount,
            loan_id=loan_id,
            borrower_address=borrower.wallet.address,
            initial_ltv=initial_loan_ltv,
            paid=False,
        )
        self._loans[loan_id] = loan
        return loan

    @running_only
    def repay(self, borrower: "CoraBorrowerAgent", loan_id: str):
        # Validate that the loan belongs to the borrower
        if borrower.wallet.address not in self._borrower_loans.keys():
            raise NonExistingBorrowerAddress()

        if loan_id not in self._borrower_loans[borrower.wallet.address]:
            raise InvalidLoanId()

        loan = self._loans[loan_id]

        # Validate that the loan has not expired
        if self._environment.get_time() > loan.expiration_time:
            raise LoanExpiredError()

        if borrower.wallet.primary_balance < loan.total_debt:
            raise InsufficientBalanceError()

        # Update balances
        self._available_amount += loan.total_debt
        self._borrowed_amount -= loan.net_loan
        self._total_collateral_locked -= loan.collateral_amount
        self._total_fees_earned += loan.borrowing_fee

        borrower.wallet.primary_balance -= loan.total_debt
        borrower.wallet.secondary_balance += loan.collateral_amount

        # Remove loan from borrower_loans
        self._borrower_loans[borrower.wallet.address].remove(loan_id)

        # Remove loan from lending pool
        self._loans[loan_id].paid = True
