import json
from pathlib import Path
from typing import Union

from simulator.utilities import DISTRIBUTIONS, BaseDistribution


class DistributionNotFound(Exception):
    def __init__(self, key: str):
        self.message = "Distribution '{}' not found".format(key)


class StrategyParamsParser:
    _distributions = DISTRIBUTIONS

    @classmethod
    def parse_json(cls, file_path: Union[str, Path]) -> dict:
        file_path = Path(file_path)
        assert file_path.exists(), f"Strategy configuration file not found: {file_path}"

        with file_path.open("r") as f:
            strategy_params: dict = json.load(f)

        return cls.parse_strategy_params(strategy_params)

    @classmethod
    def parse_strategy_params(cls, parameters: dict) -> dict:
        # Interpret distribution definitions in the json file as distributions
        parsed_parameters = {}
        for key, value in parameters.items():
            if isinstance(value, dict) and value.get("type") == "dist":
                distribution = cls.parse_distribution(value["name"], value["params"])
                parsed_parameters[key] = distribution
            else:
                parsed_parameters[key] = value
        return parsed_parameters

    @classmethod
    def parse_distribution(cls, dist_name: str, dist_params: dict) -> BaseDistribution:
        if dist_name not in cls._distributions:
            raise DistributionNotFound(dist_name)

        return cls._distributions[dist_name](**dist_params)
