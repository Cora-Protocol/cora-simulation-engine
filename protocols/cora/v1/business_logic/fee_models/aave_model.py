from datetime import timedelta
from typing import Optional

from protocols.cora.v1.business_logic.fee_models.base_fee_model import BaseCoraFeeModel
from protocols.cora.v1.environments import BaseCoraEnvironment


class AaveFeeModel(BaseCoraFeeModel):
    @staticmethod
    def get_parameters(
        environment: BaseCoraEnvironment,
        optimal_utilization: float,
        base_rate: float,
        rate_slope_1: float,
        rate_slope_2: float,
    ) -> dict:

        return {
            "optimal_utilization": optimal_utilization,
            "base_rate": base_rate,
            "rate_slope_1": rate_slope_1,
            "rate_slope_2": rate_slope_2,
        }

    def update_parameters(
        self,
        optimal_utilization: Optional[float] = None,
        base_rate: Optional[float] = None,
        rate_slope_1: Optional[float] = None,
        rate_slope_2: Optional[float] = None,
    ) -> None:
        if optimal_utilization is not None:
            self.optimal_utilization = optimal_utilization
        if base_rate is not None:
            self.base_rate = base_rate
        if rate_slope_1 is not None:
            self.rate_slope_1 = rate_slope_1
        if rate_slope_2 is not None:
            self.rate_slope_2 = rate_slope_2

    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        loan_period_years = loan_period / timedelta(days=365)

        if utilization < self.optimal_utilization:
            annualized_rate = (
                self.base_rate
                + utilization / self.optimal_utilization * self.rate_slope_1
            )
        else:
            annualized_rate = (
                self.base_rate
                + self.rate_slope_1
                + self.rate_slope_2
                * (
                    (utilization - self.optimal_utilization)
                    / (1 - self.optimal_utilization)
                )
            )

        return loan_period_years * annualized_rate
