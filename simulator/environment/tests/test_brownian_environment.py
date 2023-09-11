from datetime import datetime, timedelta
from protocols.cora.v1.environments import HistoricalCoraEnvironment
from simulator.environment.brownian_environment import BrownianSimulationEnvironment

START_DATE = datetime(2021, 1, 1)
RANGE_DAYS = 60
RANGE_DELTA = timedelta(days=RANGE_DAYS)


def test_generate_covariant_brownian_series():
    HOURS_TO_GENERATE = 11
    eth_data = (
        HistoricalCoraEnvironment("ETH")
        .set_time(START_DATE)
        .load_data_until(START_DATE + RANGE_DELTA)
        .get_price_history(RANGE_DELTA)
    )

    btc_data = (
        HistoricalCoraEnvironment("BTC")
        .set_time(START_DATE)
        .load_data_until(START_DATE + RANGE_DELTA)
        .get_price_history(RANGE_DELTA)
    )

    eth_prices = [data.price for data in eth_data]
    btc_prices = [data.price for data in btc_data]
    brownian_simulation = (
        BrownianSimulationEnvironment()._generate_brownian_continuation_from(
            [eth_prices, btc_prices], HOURS_TO_GENERATE
        )
    )

    assert (
        len(brownian_simulation[0])
        == len(brownian_simulation[1])
        == HOURS_TO_GENERATE - 1
    )
