from abc import abstractmethod
from datetime import timedelta

from protocols.cora.v1.environments import BaseCoraEnvironment


class BaseCoraFeeModel:
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def get_parameters(environment: BaseCoraEnvironment, **kwargs) -> dict:
        """Get the update parameters for the fee model based on environment status.

        Args:
            environment (BaseCoraEnvironment): The execution environment
            kwargs (dict): Additional parameters on which to base the update on

        Returns:
            dict: The updated parameters for the fee model
        """
        pass

    @abstractmethod
    def update_parameters(self, **kwargs: dict) -> None:
        pass

    @abstractmethod
    def get_fee(self, ltv: float, utilization: float, loan_period: timedelta) -> float:
        pass
