from tarfile import SUPPORTED_TYPES
from typing import Dict, List

import requests


BASE_URL = "https://api.coingecko.com/api/v3/coins/"

SUPPORTED_COINS = ["bitcoin", "ethereum", "solana", "cosmos-hub", "avalanche"]
SUPPORTED_CURRENCIES = ["usd", "eur"]


class GranularityException(Exception):
    def __init__(self):
        self.message = "This range will return daily data instead of hourly data"


class CoinNotSupported(Exception):
    def __init__(self):
        self.message = "This coin is not supported"


class CurrencyNotSupported(Exception):
    def __init__(self):
        self.message = "This currency is not supported"


def get_days_between_unix_timestamps(start: int, end: int) -> int:
    return abs((end - start) // 86400)


class CoinGeckoMarketChart:
    def __init__(self, coin_id: str, vs_currency: str = "usd"):
        if coin_id not in SUPPORTED_COINS:
            raise CoinNotSupported()

        if vs_currency not in SUPPORTED_CURRENCIES:
            raise CurrencyNotSupported()

        self.coin_id = coin_id
        self.url = f"{BASE_URL}/{coin_id}/market_chart/range?vs_currency={vs_currency}"

    def get_data(
        self, start: int, end: int, key: str = "prices"
    ) -> List[Dict[str, float]]:
        if get_days_between_unix_timestamps(start, end) > 90:
            raise GranularityException()

        # NOTE: The api only returns hourly data for time ranges longer than 1 day
        if get_days_between_unix_timestamps(start, end) < 1:
            start = start - (86400 + 1)

        response = requests.get(f"{self.url}&from={start}&to={end}")
        return response.json().get(key)
