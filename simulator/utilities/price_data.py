from typing import Dict, List, NamedTuple, NewType, Tuple, Union
from apis.coingecko.market_chart import CoinGeckoMarketChart
from simulator.utilities.data_storage import DataStorage


SYMBOL_TO_NAME = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "AVAX": "avalanche",
    "ATOM": "cosmos-hub",
}


class PriceNotFound(Exception):
    def __init__(self, timestamp: str, symbol: str):
        self.message = f"Price not found for {symbol} at {timestamp}"


class PriceDataItem(NamedTuple):
    time: int
    price: float


PriceDataRow = NewType("PriceDataItem", Dict[str, Union[int, float]])


ONE_HOUR_IN_SECONDS = 60 * 60
PRICE_DATA_FOLDER = "data"


class PriceData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_storage = DataStorage(PRICE_DATA_FOLDER)

    def get_data(self, end_time: int) -> List[PriceDataItem]:
        price_data = self.data_storage.get_csv(self.symbol)
        is_data_stale, oldest_time = self.is_data_stale(price_data, end_time)

        if is_data_stale and oldest_time < end_time:
            new_data = self.get_data_from_coingecko(oldest_time, end_time)
            price_data = [*price_data, *new_data]

        return self.get_data_up_to_timestamp(
            [
                PriceDataItem(time=int(item["time"]), price=float(item["price"]))
                for item in price_data
            ],
            end_time,
        )

    def get_data_from_coingecko(
        self, oldest_time: int, end_time: int
    ) -> List[PriceDataRow]:
        coingecko = CoinGeckoMarketChart(SYMBOL_TO_NAME.get(self.symbol, self.symbol))
        coingecko_data = coingecko.get_data(oldest_time, end_time)
        api_data = self.map_and_filter_coingecko_data(coingecko_data, oldest_time)
        self.data_storage.append_csv(self.symbol, api_data)
        return api_data

    def get_data_up_to_timestamp(
        self, data: List[PriceDataItem], timestamp: int
    ) -> List[PriceDataItem]:
        return [item for item in data if item.time <= timestamp]

    def filter_data_by_timestamps(
        self, data: List[PriceDataItem], start_time: Union[int, None], end_time: int
    ) -> List[PriceDataItem]:
        return [
            item for item in data if item.time >= start_time and item.time <= end_time
        ]

    def is_data_stale(
        self, data: List[PriceDataRow], end_time: int
    ) -> Tuple[bool, int]:
        oldest_time = self.get_oldest_time(data)
        return oldest_time < (end_time - ONE_HOUR_IN_SECONDS), oldest_time

    def get_oldest_time(self, data: List[PriceDataRow]) -> int:
        return int(data[-1]["time"])

    def get_first_time(self, data: List[PriceDataRow]) -> int:
        return int(data[0]["time"])

    def map_and_filter_coingecko_data(
        self, data: List[List[Union[float, int]]], oldest_time: int
    ) -> List[PriceDataRow]:

        return [
            {
                # NOTE: Coingecko returns timestamps in milliseconds
                "time": row[0] // 1000,
                "price": round(row[1], 1),
            }
            for row in data
            if row[0] // 1000 > oldest_time
        ]
