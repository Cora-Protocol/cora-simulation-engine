"""
This module supplies the ConstraintsConfig object and the ConstraintsConfigBuilder to allow
users of the library to configure_convolution any boundary constraints on the premium values when
generating the Kelly curve. This module is following a standard design pattern. This is the
same pattern used by the other modules of the curve generator and is mirrored here.
"""
from typing import NamedTuple, List, Callable
from dataclasses import dataclass, field


@dataclass
class ConstraintsConfig:
    """
    Immutable configuration object produced by the builder containing the boundary functions
    to use when the user wants to constrain the premium values that will be searched by the
    optimizer
    """

    upper_premium_bounds: List[Callable] = field(default_factory=list)
    lower_premium_bounds: List[Callable] = field(default_factory=list)


class ConstraintsConfigBuilder:
    """
    Standard python builder class for creating immutable configuration objects to use
    with constraining premium values when generating the Kelly curves
    """

    def __init__(self):
        """
        Constructor initializes the builder with default values
        """
        self.upper_bounds = []
        self.lower_bounds = []

    def add_upper_bound(self, boundary_fcn: Callable):
        """
        Adds a function to the list of upper bound constraints

        Parameters
        ----------
        boundary_fcn : Callable
            The function to use to constrain the upper premium values

        Returns
        -------
        self : ConstraintsConfigBuilder
            This object following the builder pattern
        """
        self.upper_bounds.append(boundary_fcn)
        return self

    def add_lower_bound(self, boundary_fcn: Callable):
        """
        Adds a function to the list of lower bound constraints

        Parameters
        ----------
        boundary_fcn : Callable
            The function to use to constrain the lower premium values

        Returns
        -------
        self : ConstraintsConfigBuilder
            This object following the builder pattern
        """
        self.lower_bounds.append(boundary_fcn)
        return self

    def build_config(self):
        """
        Creates an immutable ConstraintsConfig object from the currently configured builder

        Returns
        -------
        config : ConstraintsConfig
            The immutable configuration object
        """
        return ConstraintsConfig(self.upper_bounds, self.lower_bounds)
