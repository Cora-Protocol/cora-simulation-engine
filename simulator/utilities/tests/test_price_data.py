from datetime import datetime
from unittest import TestCase

from simulator.utilities.price_data import PriceData


class TestPriceData(TestCase):
    def test_get_price_data(self):
        price_data = PriceData("BTC")
        now = datetime.utcnow()

        data = price_data.get_data(int(now.timestamp()))
        assert int(data[-1].time) > int(now.timestamp() - 3600)

    def test_price_update(self):
        price_data = PriceData("BTC")
        now = datetime.utcnow()

        data = price_data.get_data(int(now.timestamp()))
        # assert no item in data goes back in time
        comparation_time = data[0].time
        for item in data[1:]:
            assert item.time >= comparation_time
            comparation_time = item.time
