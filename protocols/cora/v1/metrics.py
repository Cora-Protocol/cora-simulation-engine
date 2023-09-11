from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List

from protocols.cora.v1.business_logic.lending_pool import Loan
from protocols.cora.v1.environments import BaseCoraEnvironment
from protocols.cora.v1.protocol import CoraV1Protocol
from simulator.metrics import BaseSimulationMetrics, Metric
from simulator.metrics.calculations import (
    MetricBinner,
    apply_metric_binners,
    safe_divide,
)
from simulator.models.event_info import EventInfo
from simulator.models.action_info import ActionInfo
from simulator.state.state import SimulationState


class CoraMetrics(BaseSimulationMetrics):
    LOAN_SIZE_RANGES = [
        1000,
        1585,
        2512,
        3981,
        6310,
        10000,
        15849,
        25119,
        39811,
        63096,
        100000,
    ]

    @classmethod
    def by_step(cls, state: SimulationState) -> Metric:
        environment: BaseCoraEnvironment = state._environment
        protocol: CoraV1Protocol = state._protocol
        collateral_price = environment.get_price()
        lending_pools = protocol.get_lending_pools()

        if len(lending_pools) == 0:
            binned_metrics = cls._get_binned_metrics()

            return {
                "collateral_price": collateral_price,
                "pool_utilization": 0,
                "pool_liquid_capital": 0,
                "pool_capital_lent": 0,
                "active_loans_count": 0,
                "defaulted_loans_count": 0,
                "paid_loans_count": 0,
                "expired_loans_count": 0,
                "total_loans_count": 0,
                "reclaimed_collateral": 0,
                "active_loan_collateral": 0,
                "active_loans_amount": 0,
                "collateral_ratio": 0,
                "pool_realized_pnl": 0,
                "pool_unrealized_pnl": 0,
                "borrows": 0,
                "earned_fees": 0,
                "sum_of_fees": 0,
                **binned_metrics,
            }

        lending_pool = lending_pools[0]
        loans = lending_pool.get_all_loans()

        unpaid_loans = [loan for loan in loans if not loan.paid]
        expired_loans = [loan for loan in loans if loan.is_expired(environment)]

        active_loans = [l for l in unpaid_loans if not l.is_expired(environment)]
        repaid_loans = [loan for loan in expired_loans if loan.paid]
        defaulted_loans = [loan for loan in expired_loans if not loan.paid]

        pool_capital_lent_active = sum(loan.net_loan for loan in active_loans)
        pool_capital_lent_defaulted = sum(loan.net_loan for loan in defaulted_loans)
        pool_capital_lent_repaid = sum(loan.net_loan for loan in repaid_loans)

        reclaimed_collateral = sum(loan.collateral_amount for loan in defaulted_loans)
        active_loan_collateral = sum(loan.collateral_amount for loan in active_loans)

        reclaimed_collateral_value = reclaimed_collateral * collateral_price
        active_loan_collateral_value = active_loan_collateral * collateral_price

        earned_fees = sum(loan.borrowing_fee for loan in repaid_loans)

        collateral_ratio = safe_divide(
            active_loan_collateral_value, pool_capital_lent_active
        )

        pool_realized_pnl = (
            earned_fees + reclaimed_collateral_value - pool_capital_lent_defaulted
        )

        pool_unrealized_pnl = (
            sum(
                min(collateral_price * loan.collateral_amount, loan.total_debt)
                - loan.net_loan
                for loan in active_loans
            )
            + pool_realized_pnl
        )

        run_delta = lending_pool._running_period
        run_end = lending_pool._next_cycle_time
        binned_metrics = cls._get_binned_metrics(
            active_loans, defaulted_loans, repaid_loans, loans, run_delta, run_end
        )

        return {
            "collateral_price": collateral_price,
            "pool_utilization": lending_pool.get_current_utilization(),
            "pool_liquid_capital": lending_pool.get_current_available_amount(),
            "pool_capital_lent": pool_capital_lent_active,
            "active_loans_count": len(active_loans),
            "defaulted_loans_count": len(defaulted_loans),
            "paid_loans_count": len(repaid_loans),
            "expired_loans_count": len(expired_loans),
            "total_loans_count": len(loans),
            "reclaimed_collateral": reclaimed_collateral,
            "active_loan_collateral": active_loan_collateral,
            "active_loans_amount": pool_capital_lent_active,
            "collateral_ratio": collateral_ratio,
            "pool_realized_pnl": pool_realized_pnl,
            "pool_unrealized_pnl": pool_unrealized_pnl,
            "borrows": len(loans),
            "earned_fees": earned_fees,
            "sum_of_fees": sum(loan.borrowing_fee for loan in loans),
            **binned_metrics,
        }

    @classmethod
    def custom_events(
        cls,
        state: SimulationState,
        actions_info: List[ActionInfo],
        events_info: List[EventInfo],
    ) -> Dict[str, List[dict]]:
        metrics = {}
        for event in events_info:
            if event.type == "lending_pool_running_period_ended":
                lending_pool_name: str = event.extra["lending_pool"]
                cycle_number: int = event.extra["cycle_number"]

                protocol: CoraV1Protocol = state._protocol
                lending_pool = protocol.get_lending_pool(lending_pool_name)
                cycle_metrics = lending_pool._cycle_history[cycle_number]
                cycle_loans = cycle_metrics.loans
                run_delta = lending_pool._running_period
                run_end = lending_pool._next_cycle_time - run_delta

                initial_liquidity = cycle_metrics.initial_liquidity
                final_liquidity = cycle_metrics.remaining_liquidity
                total_earned_fees = cycle_metrics.total_fees_earned
                total_reclaimed_collateral = cycle_metrics.total_reclaimed_collateral
                collateral_value = cycle_metrics.final_collateral_value

                pnl = final_liquidity + collateral_value - initial_liquidity
                liquidity_change = final_liquidity - initial_liquidity

                pnl_ratio = safe_divide(pnl, initial_liquidity)
                liquidity_change_ratio = safe_divide(
                    liquidity_change, initial_liquidity
                )

                repaid_loans = [loan for loan in cycle_loans if loan.paid]
                defaulted_loans = [loan for loan in cycle_loans if not loan.paid]

                binned_metrics = cls._get_binned_metrics(
                    [], defaulted_loans, repaid_loans, cycle_loans, run_delta, run_end
                )

                if "cycle_end" not in metrics:
                    metrics["cycle_end"] = []
                metrics["cycle_end"].append(
                    {
                        "lending_pool": lending_pool_name,
                        "cycle_number": cycle_number,
                        "pnl": pnl,
                        "liquidity_change": liquidity_change,
                        "pnl_ratio": pnl_ratio,
                        "liquidity_change_ratio": liquidity_change_ratio,
                        "total_earned_fees": total_earned_fees,
                        "total_reclaimed_collateral": total_reclaimed_collateral,
                        "collateral_value": collateral_value,
                        "initial_liquidity": initial_liquidity,
                        "final_liquidity": final_liquidity,
                        "average_utilization": cycle_metrics.average_utilization,
                        "normalized_utilization": cycle_metrics.normalized_utilization,
                        "num_lans": len(cycle_loans),
                        **binned_metrics,
                    }
                )
        return metrics

    @classmethod
    def end_of_simulation(cls, state: SimulationState, metrics: List[Metric]) -> Metric:
        environment: BaseCoraEnvironment = state._environment
        protocol: CoraV1Protocol = state._protocol
        collateral_price = environment.get_price()

        lending_pools = protocol.get_lending_pools()
        if len(lending_pools) == 0:
            binned_metrics = cls._get_binned_metrics()
            return {
                "pool_pnl": 0,
                "ratio_loans_defaulted": 0,
                "lending_fees": 0,
                **binned_metrics,
            }

        lending_pool = lending_pools[0]
        loans = lending_pool.get_all_loans()

        unpaid_loans = [loan for loan in loans if not loan.paid]
        expired_loans = [loan for loan in loans if loan.is_expired(environment)]

        active_loans = [l for l in unpaid_loans if not l.is_expired(environment)]
        repaid_loans = [loan for loan in expired_loans if loan.paid]
        defaulted_loans = [loan for loan in expired_loans if not loan.paid]

        ratio_loans_defaulted = safe_divide(len(defaulted_loans), len(loans))

        pool_capital_lent_active = sum(loan.net_loan for loan in active_loans)
        pool_capital_lent_defaulted = sum(loan.net_loan for loan in defaulted_loans)
        pool_capital_lent_repaid = sum(loan.net_loan for loan in repaid_loans)

        reclaimed_collateral = sum(loan.collateral_amount for loan in defaulted_loans)

        pool_capital_lent = sum(loan.net_loan for loan in active_loans)
        pool_capital_lent_unpaid = sum(loan.net_loan for loan in unpaid_loans)
        reclaimed_collateral = sum([loan.collateral_amount for loan in unpaid_loans])
        collateral_price = environment.get_price()
        pool_realized_pnl = (
            sum(loan.borrowing_fee for loan in repaid_loans)
            + reclaimed_collateral * collateral_price
            - pool_capital_lent_unpaid
        )

        run_delta = lending_pool._running_period
        run_end = lending_pool._next_cycle_time
        binned_metrics = cls._get_binned_metrics(
            active_loans, defaulted_loans, repaid_loans, loans, run_delta, run_end
        )

        return {
            "pool_pnl": pool_realized_pnl,
            "ratio_loans_defaulted": defaulted_loans,
            "lending_fees": sum((loan.borrowing_fee for loan in loans)),
            **binned_metrics,
        }

    @classmethod
    def _get_binned_metrics(
        cls,
        active_loans: Iterable[Loan] = tuple(),
        defaulted_loans: Iterable[Loan] = tuple(),
        repaid_loans: Iterable[Loan] = tuple(),
        loans: Iterable[Loan] = tuple(),
        run_delta: timedelta = None,
        run_end: datetime = None,
    ) -> Dict[str, float]:
        loan_binners: List[MetricBinner] = [
            MetricBinner("ltv", lambda loan: loan.initial_ltv),
            MetricBinner("duration", lambda loan: loan.get_duration() / run_delta),
            MetricBinner("start", lambda l: 1 - (run_end - l.start_time) / run_delta),
            MetricBinner("size", lambda loan: loan.net_loan, cls.LOAN_SIZE_RANGES),
        ]
        active_loans_hists = apply_metric_binners(
            active_loans, loan_binners, "hist-active_loans"
        )
        defaulted_loans_hists = apply_metric_binners(
            defaulted_loans, loan_binners, "hist-defaulted_loans"
        )
        repaid_loans_hists = apply_metric_binners(
            repaid_loans, loan_binners, "hist-repaid_loans"
        )
        loans_hists = apply_metric_binners(loans, loan_binners, "hist-loans")
        loan_fees_dists = apply_metric_binners(
            loans,
            loan_binners,
            "dist-loan_fees",
            lambda ls: sum(l.borrowing_fee for l in ls),
        )
        return {
            **active_loans_hists,
            **defaulted_loans_hists,
            **repaid_loans_hists,
            **loans_hists,
            **loan_fees_dists,
        }
