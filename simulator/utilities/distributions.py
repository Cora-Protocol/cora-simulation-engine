from abc import abstractmethod

import numpy as np
from scipy.stats import norm, powerlaw, triang, truncnorm, uniform


class BaseDistribution:
    def __init__(self):
        self._rng = np.random.default_rng()

    @abstractmethod
    def sample(self) -> float:
        pass

    def set_rng(self, rng: np.random.Generator):
        self._rng = rng


class MockDistribution(BaseDistribution):
    def sample(self):
        return self._rng.uniform(0, 1)


class UniformDistribution(BaseDistribution):
    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        super().__init__()
        self.frozen = uniform(lower, upper - lower)

    def sample(self) -> float:
        return self.frozen.rvs(random_state=self._rng)


class NormalDistribution(BaseDistribution):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.frozen = norm(mean, std)

    def sample(self) -> float:
        return self.frozen.rvs(random_state=self._rng)


class TriangularDistribution(BaseDistribution):
    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        super().__init__()
        if upper > lower:
            self.reverse = False
            self.frozen = triang(1.0, loc=lower, scale=(upper - lower))
        else:
            self.reverse = True
            self.frozen = triang(1.0, loc=-lower, scale=(lower - upper))

    def sample(self) -> float:
        sampled_num = self.frozen.rvs(random_state=self._rng)
        return -sampled_num if self.reverse else sampled_num


class ParabolicDistribution(BaseDistribution):
    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        super().__init__()
        if upper > lower:
            self.reverse = False
            self.frozen = powerlaw(3.0, loc=lower, scale=(upper - lower))
        else:
            self.reverse = True
            self.frozen = powerlaw(3.0, loc=-lower, scale=(lower - upper))

    def sample(self) -> float:
        sampled_num = self.frozen.rvs(random_state=self._rng)
        return -sampled_num if self.reverse else sampled_num


class TruncatedNormalDistribution(BaseDistribution):
    def __init__(self, lower: float, upper: float, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        a, b = (lower - mean) / std, (upper - mean) / std
        self.frozen = truncnorm(a, b, mean, std)

    def sample(self) -> float:
        return self.frozen.rvs(random_state=self._rng)


class TruncatedInverseNormalDistribution(BaseDistribution):
    def __init__(self, lower: float, upper: float, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        inverse_low = 1.0 / upper if upper != 0 else mean - std * 10.0
        inverse_up = 1.0 / lower if lower != 0 else mean + std * 10.0
        a, b = (inverse_low - mean) / std, (inverse_up - mean) / std
        self.frozen = truncnorm(a, b, mean, std)

    def sample(self) -> float:
        norm_sample = self.frozen.rvs(random_state=self._rng)
        return 1.0 / norm_sample


class LogNormalDistribution(BaseDistribution):
    def __init__(self, mean: float = 0.0, std: float = 1.0, base: float = np.e):
        super().__init__()
        self._base = base
        self._expfunc = lambda x: base**x
        self.frozen = norm(mean, std)

    def sample(self) -> float:
        return self._expfunc(self.frozen.rvs(random_state=self._rng))


class TruncatedLogNormalDistribution(BaseDistribution):
    def __init__(
        self,
        lower: float,
        upper: float,
        mean: float = 0.0,
        std: float = 1.0,
        base: float = np.e,
    ):
        super().__init__()
        self._base = base
        self._logfunc = lambda x: np.log(x) / np.log(base)
        self._expfunc = lambda x: base**x
        a = (self._logfunc(lower) - mean) / std if lower != 0 else mean - std * 10.0
        b = (self._logfunc(upper) - mean) / std if upper != 0 else mean + std * 10.0
        self.frozen = truncnorm(a, b, mean, std)

    def sample(self) -> float:
        return self._expfunc(self.frozen.rvs(random_state=self._rng))
