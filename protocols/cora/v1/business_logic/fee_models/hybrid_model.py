from abc import abstractmethod
from datetime import timedelta
from typing import Optional, List, Dict

from protocols.cora.v1.business_logic.fee_models import (
    AaveFeeModel,
    BlackScholesFeeModel,
    CachedKellyFeeModel,
)
from protocols.cora.v1.business_logic.fee_models.base_fee_model import BaseCoraFeeModel
from protocols.cora.v1.business_logic.fee_models.black_scholes_model import (
    UtilizationCurve,
)
from protocols.cora.v1.environments import BaseCoraEnvironment
from protocols.cora.v1.business_logic.fee_models.kelly_fee_model import (
    KellyCurve,
    GridPoint,
)


class HybricBlackScholesAave(BaseCoraFeeModel):
    def __init__(self):
        self.black_scholes = BlackScholesFeeModel()
        self.aave = AaveFeeModel()

    @staticmethod
    def get_parameters(
        environment: BaseCoraEnvironment,
        lookback_days: int,
        volatility_factor: float,
        optimal_utilization: float,
        base_rate: float,
        rate_slope_1: float,
        rate_slope_2: float,
    ) -> dict:
        black_scholes_params = BlackScholesFeeModel.get_parameters(
            environment, lookback_days, volatility_factor
        )
        aave_params = AaveFeeModel.get_parameters(
            environment, optimal_utilization, base_rate, rate_slope_1, rate_slope_2
        )

        return {
            **black_scholes_params,
            **aave_params,
        }

    def update_parameters(
        self,
        volatility: Optional[float] = None,
        risk_free_rate: Optional[float] = None,
        utilization_curve: Optional[UtilizationCurve] = None,
        optimal_utilization: Optional[float] = None,
        base_rate: Optional[float] = None,
        rate_slope_1: Optional[float] = None,
        rate_slope_2: Optional[float] = None,
    ) -> None:
        self.black_scholes.update_parameters(
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            utilization_curve=utilization_curve,
        )
        self.aave.update_parameters(
            optimal_utilization=optimal_utilization,
            base_rate=base_rate,
            rate_slope_1=rate_slope_1,
            rate_slope_2=rate_slope_2,
        )

    @abstractmethod
    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        pass


class SumBlackScholesAave(HybricBlackScholesAave):
    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        black_scholes_fee = self.black_scholes.get_fee(ltv, utilization, loan_period)
        aave_fee = self.aave.get_fee(ltv, utilization, loan_period)
        return black_scholes_fee + aave_fee


class CombinedBlackScholesAave(HybricBlackScholesAave):
    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        black_scholes_fee = self.black_scholes.get_fee(ltv, utilization, loan_period)
        aave_fee = self.aave.get_fee(ltv, utilization, loan_period)
        if aave_fee > black_scholes_fee:
            return 0.5 * (aave_fee + black_scholes_fee)
        else:
            return black_scholes_fee


class HybridKellyAave(BaseCoraFeeModel):
    def __init__(self):
        self.kelly = CachedKellyFeeModel()
        self.aave = AaveFeeModel()

    def get_parameters(
        self,
        environment: BaseCoraEnvironment,
        lookback_days: int,
        ltv_values: List[float],
        max_expiration_days: int,
        interval_days: int,
        optimal_utilization: float,
        base_rate: float,
        rate_slope_1: float,
        rate_slope_2: float,
    ) -> dict:
        black_scholes_params = self.kelly.get_parameters(
            environment,
            lookback_days,
            ltv_values,
            max_expiration_days,
            interval_days,
        )
        aave_params = self.aave.get_parameters(
            environment, optimal_utilization, base_rate, rate_slope_1, rate_slope_2
        )

        return {
            **black_scholes_params,
            **aave_params,
        }

    def update_parameters(
        self,
        curve_grid: Optional[Dict[GridPoint, KellyCurve]] = None,
        risk_free_rate: Optional[float] = None,
        utilization_curve: Optional[UtilizationCurve] = None,
        optimal_utilization: Optional[float] = None,
        base_rate: Optional[float] = None,
        rate_slope_1: Optional[float] = None,
        rate_slope_2: Optional[float] = None,
    ) -> None:
        self.kelly.update_parameters(
            curve_grid,
        )
        self.aave.update_parameters(
            optimal_utilization=optimal_utilization,
            base_rate=base_rate,
            rate_slope_1=rate_slope_1,
            rate_slope_2=rate_slope_2,
        )

    @abstractmethod
    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        pass


class SumKellyAave(HybridKellyAave):
    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        kelly_fee = self.kelly.get_fee(ltv, utilization, loan_period)
        aave_fee = self.aave.get_fee(ltv, utilization, loan_period)
        return kelly_fee + aave_fee


class CombinedKellyAave(HybridKellyAave):
    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        kelly_fee = self.kelly.get_fee(ltv, utilization, loan_period)
        aave_fee = self.aave.get_fee(ltv, utilization, loan_period)
        if aave_fee > kelly_fee:
            return 0.5 * (aave_fee + kelly_fee)
        else:
            return kelly_fee
