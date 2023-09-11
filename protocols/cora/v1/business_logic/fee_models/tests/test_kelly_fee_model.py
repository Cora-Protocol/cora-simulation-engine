from datetime import datetime

from protocols.cora.v1.environments import HistoricalCoraEnvironment
from protocols.cora.v1.business_logic.fee_models.kelly_fee_model import (
    KellyCurve,
    KellyFeeModel,
)

parameter_getter = KellyFeeModel.get_parameters


def test_parameter_getter():
    environment = HistoricalCoraEnvironment("ETH")
    environment.set_time(datetime(2022, 7, 1))
    environment.load_data_until(datetime(2022, 7, 1))
    parameters = parameter_getter(
        environment,
        lookback_days=365,
        ltv_values=[0.7, 0.8],
        max_expiration_days=30,
        interval_days=20,
    )
    curve_grid = parameters["curve_grid"]
    kelly_curve_result = curve_grid[(0.8, 30)]
    # NOTE: Change this to use math.isclose?
    assert kelly_curve_result == KellyCurve(
        0.005580790160052604,
        0.7632373408055823,
        0.7228136048135454,
        0.028376116013514155,
    )
