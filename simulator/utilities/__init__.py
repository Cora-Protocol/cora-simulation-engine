from simulator.utilities.distributions import (
    BaseDistribution,
    LogNormalDistribution,
    MockDistribution,
    NormalDistribution,
    ParabolicDistribution,
    TriangularDistribution,
    TruncatedInverseNormalDistribution,
    TruncatedLogNormalDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
)

DISTRIBUTIONS: dict[str, BaseDistribution] = {
    "mock": MockDistribution,
    "normal": NormalDistribution,
    "lognormal": LogNormalDistribution,
    "parabolic": ParabolicDistribution,
    "triangular": TriangularDistribution,
    "truncated_inverse_normal": TruncatedInverseNormalDistribution,
    "truncated_lognormal": TruncatedLogNormalDistribution,
    "truncated_normal": TruncatedNormalDistribution,
    "uniform": UniformDistribution,
}
