import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
from libs.curve_gen.gen import Generator
from libs.curve_gen.training.builder import CurveConfig
from libs.curve_gen.utils import build_generator_config
from numpy.typing import ArrayLike
from protocols.cora.v1.business_logic.fee_models.base_fee_model import BaseCoraFeeModel
from protocols.cora.v1.environments import BaseCoraEnvironment


class GridPoint(NamedTuple):
    ltv: float
    days: float


@dataclass
class KellyCurve:
    a: float
    b: float
    c: float
    d: float

    def evaluate(self, utilization: float) -> float:
        if utilization > 1.0 or utilization < 0.0:
            raise ValueError(f"utilization must be between 0 and 1.0: {utilization}")
        return self.a * utilization * np.cosh(self.b * utilization**self.c) + self.d


def select_next_highest(sorted_array: ArrayLike, value: float) -> float:
    """Select the next highest value in the sorted array, or the max if it's lower.

    Args:
        sorted_array (ArrayLike): The sorted array to search
        value (float): The value to search for

    Returns:
        float: The next highest value in the sorted array
    """
    idx = max(np.searchsorted(sorted_array, value, side="right"), len(sorted_array) - 1)
    return sorted_array[idx]


class KellyFeeModel(BaseCoraFeeModel):
    @staticmethod
    def get_parameters(
        environment: BaseCoraEnvironment,
        lookback_days: int,
        ltv_values: List[float],
        max_expiration_days: int,
        interval_days: int = 1,
    ) -> dict:

        time = environment.get_time()
        time_date = datetime(time.year, time.month, time.day)
        start_datetime = time_date - timedelta(days=lookback_days)
        dates = [start_datetime + timedelta(days=i) for i in range(lookback_days + 1)]
        timestamps = [int(date.timestamp()) for date in dates]

        prices = environment.get_price_for_timestamps(timestamps)
        current_price = prices[-1]

        price_history = [(date, price) for date, price in zip(dates, prices)]

        expiration_days = list(
            range(interval_days, max_expiration_days + 1, interval_days)
        )
        if max_expiration_days not in expiration_days:
            expiration_days.append(max_expiration_days)

        curve_configs = [
            CurveConfig(
                asset="placeholder",
                training_start=start_datetime,
                training_end=time_date,
                strike_percent=ltv,
                expiration=expiration,
                current_price=current_price,
            )
            for ltv, expiration in product(ltv_values, expiration_days)
        ]

        generator_config = build_generator_config(
            configs=curve_configs,
            price_history=price_history,
        )
        gen = Generator()
        gen.configure_curve_gen(generator_config)

        curve_df, _, _ = gen.generate_curves()
        curve_grid = {
            GridPoint(row["StrikePercent"], row["Expiration"]): KellyCurve(
                a=row["A"], b=row["B"], c=row["C"], d=row["D"]
            )
            for i, row in curve_df.iterrows()
        }
        return {"curve_grid": curve_grid}

    def update_parameters(self, curve_grid: Dict[GridPoint, KellyCurve]) -> None:
        self.curve_grid = curve_grid
        self.ltvs = np.array(sorted(list({point.ltv for point in curve_grid})))
        self.days = np.array(sorted(list({point.days for point in curve_grid})))
        assert len(self.curve_grid) == len(self.ltvs) * len(self.days)

    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        # Select closest most conservative curve from grid
        grid_ltv = select_next_highest(self.ltvs, ltv)
        grid_days = select_next_highest(self.days, loan_period.days)
        curve = self.curve_grid[GridPoint(ltv=grid_ltv, days=grid_days)]

        # Evaluate curve
        return curve.evaluate(utilization)


class CachedKellyFeeModel(KellyFeeModel):
    """Only considers a single update at the start of the simulation, and tries to
    retrieve the curve from the cache.
    """

    CACHE_LOCATION = Path("protocols/cora/v1/business_logic/fee_models/kelly_cache")

    def __init__(self):
        super().__init__()
        self.CACHE_LOCATION.mkdir(parents=True, exist_ok=True)
        self.cache = None

    def get_parameters(
        self,
        environment: BaseCoraEnvironment,
        lookback_days: int,
        ltv_values: List[float],
        max_expiration_days: int,
        interval_days: int = 1,
    ) -> dict:
        # if cache is already loaded, return that
        if self.cache is not None:
            return self.cache

        # check if cache exists as file
        initial_date = environment.get_time()
        initial_date_str = initial_date.strftime("%Y-%m-%d")
        file_name = f"{initial_date_str}_lb{lookback_days}_exp{max_expiration_days}_kelly_fee_model.pkl"
        cache_path = self.CACHE_LOCATION / file_name

        if cache_path.exists():
            with cache_path.open("rb") as f:
                self.cache: Dict[GridPoint, KellyCurve] = pickle.load(f)
            return self.cache

        # otherwise, generate and save cache
        print(f"No cache found for {file_name}, generating...")
        parameters = super().get_parameters(
            environment,
            lookback_days,
            ltv_values,
            max_expiration_days,
            interval_days,
        )
        print(f"Calculated cache for {file_name}")
        with cache_path.open("wb") as f:
            pickle.dump(parameters, f)
        print(f"Generated cache for {file_name}")
        self.cache = parameters
        return parameters
