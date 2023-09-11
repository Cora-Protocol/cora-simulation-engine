from datetime import datetime, timedelta
from unittest import TestCase

import pytest

from apis.coingecko.market_chart import (
    CoinGeckoMarketChart,
    CoinNotSupported,
    CurrencyNotSupported,
    GranularityException,
    get_days_between_unix_timestamps,
)


class TestMarketChart(TestCase):
    def test_calculate_distance_between_timestamps(self):
        excepted_days = 2
        now = datetime.utcnow()
        two_days_ago = now - timedelta(days=excepted_days)
        days = get_days_between_unix_timestamps(
            int(now.timestamp()), int(two_days_ago.timestamp())
        )
        assert days == excepted_days

    def test_supported_coins(self):
        with pytest.raises(CoinNotSupported):
            CoinGeckoMarketChart("not_supported")

        assert CoinGeckoMarketChart("bitcoin").coin_id == "bitcoin"

    def test_supported_currency(self):
        with pytest.raises(CurrencyNotSupported):
            CoinGeckoMarketChart("bitcoin", "not_supported")

        assert CoinGeckoMarketChart("bitcoin", "usd").coin_id == "bitcoin"

    def test_get_data_fails_for_more_than_90_days(self):
        api = CoinGeckoMarketChart("bitcoin", "usd")
        now = datetime.utcnow()
        hundred_days_ago = now - timedelta(days=100)
        with pytest.raises(GranularityException):
            api.get_data(int(hundred_days_ago.timestamp()), int(now.timestamp()))

    def test_get_data_works(self):
        api = CoinGeckoMarketChart("bitcoin", "usd")
        now = datetime.utcnow()
        two_days_ago = now - timedelta(days=2)
        data = api.get_data(int(two_days_ago.timestamp()), int(now.timestamp()))
        assert len(data) > 0
