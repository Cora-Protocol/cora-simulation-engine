from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np

from protocols.cora.v1.environments import (
    BrownianCoraEnvironment,
    HistoricalCoraEnvironment,
)

START_DATE = datetime(2021, 1, 1)
RANGE_DAYS = 360
TEST_DAYS = 180
RANGE_DELTA = timedelta(days=RANGE_DAYS)
TEST_DELTA = timedelta(days=TEST_DAYS)


class TestHistoricalEnvironment(TestCase):
    def test_loads_data(self):
        environment = HistoricalCoraEnvironment("ETH")
        environment.load_data_until(START_DATE + RANGE_DELTA)

        assert len(environment.get_price_data()) > RANGE_DAYS * 24 + 1

    def test_get_price_for_multiple_timestamps(self):
        environment = HistoricalCoraEnvironment("ETH")
        environment.load_data_until(START_DATE + RANGE_DELTA)

        result = environment.get_price_for_timestamps(
            [
                (START_DATE + TEST_DELTA).timestamp(),
                (START_DATE + TEST_DELTA + timedelta(hours=1)).timestamp(),
                (START_DATE + TEST_DELTA + timedelta(hours=2)).timestamp(),
            ]
        )
        assert len(result) == 3

    def test_get_price_for_timestamp_searching(self):
        environment = HistoricalCoraEnvironment("ETH").load_data_until(
            START_DATE + RANGE_DELTA
        )

        result = environment.get_searchsorted_price_for_timestamp(
            (START_DATE + TEST_DELTA).timestamp()
        )

        assert result > 0

    def test_get_price_history(self):
        environment = (
            HistoricalCoraEnvironment("ETH")
            .load_data_until(START_DATE + RANGE_DELTA)
            .set_time(START_DATE + TEST_DELTA - timedelta(hours=2))
        )

        result = environment.get_price_history(TEST_DELTA)
        assert len(result) == TEST_DAYS * 24


class TestBrownianEnvironment(TestCase):
    def test_loads_data(self):
        environment = (
            BrownianCoraEnvironment("ETH")
            .set_time(START_DATE + TEST_DELTA - timedelta(hours=2))
            .set_rng(np.random.default_rng(200))
            .load_data_until(START_DATE + RANGE_DELTA)
        )
        data = environment.get_price_data()
        assert len(data) > RANGE_DAYS * 24 + 1
        assert len(set((item.time for item in data))) == len(data)

    def test_get_price_for_multiple_timestamps(self):
        environment = (
            BrownianCoraEnvironment("ETH")
            .set_time(START_DATE + TEST_DELTA - timedelta(hours=2))
            .set_rng(np.random.default_rng(200))
            .load_data_until(START_DATE + RANGE_DELTA)
        )
        result = environment.get_price_for_timestamps(
            [
                (START_DATE + TEST_DELTA).timestamp(),
                (START_DATE + TEST_DELTA + timedelta(hours=1)).timestamp(),
                (START_DATE + TEST_DELTA + timedelta(hours=2)).timestamp(),
            ]
        )
        assert len(result) == 3

    def test_get_price_for_timestamp_searching(self):
        environment = (
            BrownianCoraEnvironment("ETH")
            .set_time(START_DATE + TEST_DELTA - timedelta(hours=2))
            .set_rng(np.random.default_rng(200))
            .load_data_until(START_DATE + RANGE_DELTA)
        )

        result = environment.get_searchsorted_price_for_timestamp(
            (START_DATE + TEST_DELTA).timestamp()
        )

        assert result > 0

    def test_get_price_history(self):
        environment = (
            BrownianCoraEnvironment("ETH")
            .set_time(START_DATE + TEST_DELTA - timedelta(hours=2))
            .set_rng(np.random.default_rng(200))
            .load_data_until(START_DATE + RANGE_DELTA)
        )

        result = environment.get_price_history(TEST_DELTA)
        assert len(result) == TEST_DAYS * 24

    def test_different_rngs(self):
        first_env = (
            BrownianCoraEnvironment("ETH")
            .set_rng(np.random.default_rng(seed=200))
            .set_time(START_DATE)
            .load_data_until(START_DATE + RANGE_DELTA)
            .set_time(START_DATE + RANGE_DELTA)
        )
        first_data = first_env.get_price_history(TEST_DELTA)

        second_env = (
            BrownianCoraEnvironment("ETH")
            .set_rng(np.random.default_rng(seed=34567890))
            .set_time(START_DATE)
            .load_data_until(START_DATE + RANGE_DELTA)
            .set_time(START_DATE + RANGE_DELTA)
        )
        second_data = second_env.get_price_history(TEST_DELTA)

        assert first_data[-1] != second_data[-1]
