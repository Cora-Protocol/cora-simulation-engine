import sys
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from protocols.cora.v1.business_logic.fee_models.base_fee_model import BaseCoraFeeModel
from scipy.stats import norm

from protocols.cora.v1.environments import BaseCoraEnvironment
from math import sqrt

UtilizationCurve = Callable[[float], float]


def check_div_zero(value, close_value=1e-10):
    """
    Checks if the value passed is zero. If the value is zero, it is replaced with a
    float that is extremely close to zero. When doing lots of calculations, this is
    faster than using the exception stack to check for ZeroDivisionError

    Parameters
    ----------
    value : float or numpy.ndarray
        The value to check if it is zero
    close_value : float
        The value to replace with. Cannot be 0.0. (Default: 1e-30)

    Returns
    -------
    value : float or numpy.ndarray
        The value with zero replaced
    """

    if isinstance(value, np.ndarray):
        value[value == 0.0] = close_value
    elif value == 0.0:
        value = close_value

    return value


def _d1(s, r: float, q: float, k: float, tau: float, sigma: float):
    """
    Calculates the d1 parameter of the Black-Scholes equation (sometimes called d+)

    Parameters
    ----------
    s : float
        The current price of the underlying asset
    r : float
        The interest annualized rate in the currency which the option was struck
    q : float
        The rate of annualized return of cash flow from the underlying (like a dividend)
    k : float
        The strike price of the option
    tau : float
        The time in years from now until the option expiration
    sigma : float
        The volatility of the underlying asset

    Returns
    -------
    d1 : float
        The output of the d1 calculation
    """
    return (np.log(s / k) + ((r - q) + (sigma * sigma / 2.0)) * tau) / (
        sigma * np.sqrt(tau)
    )


def _d2(d_1, sigma: float, tau: float):
    """
    Calculates the d2 parameter of the Black-Scholes equation (sometimes called d-)

    Parameters
    ----------
    d_1 : float
        The output of the d1 calculation so that it is not calculated twice
    sigma : float
        The volatility of the underlying asset
    tau : float
        The time in years from now until the option expiration

    Returns
    -------
    d2 : float
        The output of the d2 calculation
    """
    return d_1 - sigma * np.sqrt(tau)


def put_premium(
    s: Union[float, np.ndarray],
    k: Union[float, np.ndarray],
    tau: float,
    sigma: Union[float, np.ndarray],
    r: float,
    q: float,
) -> Union[float, np.ndarray]:
    """
    Calculates the price of the Put and optionally all of the Greeks associated with the
    option and returns the price and a dict containing the greeks.

    Parameters
    ----------
    s : Union[float, np.ndarray]
        The current price of the underlying asset
    k : Union[float, np.ndarray]
        The strike price of the option
    tau : float
        The time in years from now until the option expiration
    sigma : Union[float, np.ndarray]
        The volatility of the underlying asset
    r : float
        The interest annualized return in the currency which the option was struck
    q : float
        The rate of annualized return of cash flow from the underlying (like a dividend)

    Returns
    -------
    put_price : float
        The option price
    """
    # If parameter is 0 make it just above 0 so we don't get NaNs
    s = check_div_zero(s)
    k = check_div_zero(k)
    sigma = check_div_zero(sigma)
    tau = check_div_zero(tau)

    # Calculate the d values for the equation
    d1 = _d1(s, r, q, k, tau, sigma)
    d2 = _d2(d1, sigma, tau)

    # Calculate the price
    strike_term = k * np.exp(-r * tau) * norm.cdf(-d2)
    price_term = s * np.exp(-q * tau) * norm.cdf(-d1)

    put_price = strike_term - price_term
    return put_price


class BlackScholesFeeModel(BaseCoraFeeModel):
    @staticmethod
    def get_parameters(
        environment: BaseCoraEnvironment,
        lookback_days: int,
        volatility_factor: float,
    ) -> dict:
        # NOTE: Would be useful to move the RVol calculation to a different module
        price_history = environment.get_price_history(timedelta(days=lookback_days))
        assert len(price_history) > 1, "Price history must be at least 2 periods long"

        times, prices = zip(*price_history)

        log_returns: np.ndarray = np.log(prices[1:]) - np.log(prices[:-1])
        var = np.sum((log_returns**2.0)).item()

        n_returns = len(log_returns)
        n_years = timedelta(seconds=(times[-1] - times[0])) / timedelta(days=365)
        periods_in_year = n_returns / n_years

        volatility = sqrt(periods_in_year / n_returns * var) * volatility_factor

        return {
            "volatility": volatility,
            "risk_free_rate": 0.0,
            "utilization_curve": lambda x: 1.0,
        }

    def update_parameters(
        self,
        volatility: Optional[float] = None,
        risk_free_rate: Optional[float] = None,
        utilization_curve: Optional[UtilizationCurve] = None,
    ) -> None:
        if volatility is not None:
            self._update_volatility(volatility)
        if risk_free_rate is not None:
            self._update_risk_free_rate(risk_free_rate)
        if utilization_curve is not None:
            self._update_utilization_curve(utilization_curve)

    def _update_volatility(self, volatility: float) -> None:
        self.volatility = volatility

    def _update_risk_free_rate(self, risk_free_rate: float) -> None:
        self.risk_free_rate = risk_free_rate

    def _update_utilization_curve(self, utilization_curve: UtilizationCurve) -> None:
        self.utilization_curve = utilization_curve

    def _utilization_factor(self, utilization: float) -> float:
        return self.utilization_curve(utilization)

    def _option_premium(self, ltv: float, loan_period: timedelta) -> float:
        tau = loan_period / timedelta(days=365)
        return put_premium(1.0, ltv, tau, self.volatility, self.risk_free_rate, 0.0)

    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        price = self._option_premium(ltv, loan_period)
        utilization_factor = self._utilization_factor(utilization)
        return price * utilization_factor
