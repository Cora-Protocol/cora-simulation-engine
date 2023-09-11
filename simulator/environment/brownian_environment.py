from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import multivariate_normal, norm
from simulator.environment.base_environment import BaseSimulationEnvironment


class BrownianSimulationEnvironment(BaseSimulationEnvironment):
    def _multivariate_geometric_brownian_returns(
        self,
        mu: ArrayLike,
        sigma: ArrayLike,
        corr: ArrayLike,
        dt: float,
        T: int,
        sigma_factor: float = 1.0,
        rng=None,
    ) -> np.ndarray:
        """Generate multivariate geometric Brownian motion returns with correlation

        Args:
            mu (ArrayLike): means of log returns, size N
            sigma (ArrayLike): volatilities of log returns, size N
            corr (ArrayLike): correlation matrix of log returns, size N x N
            dt (float): time step
            T (int): number of values to generate
            rng (Optional[np.random.RandomState], optional): Random number generator.

        Returns:
            ArrayLike: multivariate geometric Brownian motion returns of shape (N, T)
        """
        mu = np.array(mu).flatten().astype(float).reshape(-1, 1)
        sigma = np.array(sigma).flatten().astype(float).reshape(-1, 1)
        corr = np.array(corr).astype(float)

        assert mu.shape == sigma.shape, "mu and sigma must have the same length"
        assert mu.shape[0] == corr.shape[0], "mu and corr must have the same length"
        assert corr.shape[0] == corr.shape[1], "Correlation matrix must be square"
        assert len(corr.shape) == 2, "corr must be a 2D array"

        if len(mu) == 1:
            epsilon = norm.rvs(size=T, random_state=rng).transpose()
        else:
            epsilon = multivariate_normal.rvs(
                cov=corr, size=T, random_state=rng
            ).transpose()
        stochastic = epsilon * np.sqrt(dt) * sigma * sigma_factor
        drift = (mu - (sigma * sigma_factor) ** 2 / 2) * dt
        return np.exp(drift + stochastic)

    def _multivariate_geometric_brownian_series(
        self,
        Si: ArrayLike,
        mu: ArrayLike,
        sigma: ArrayLike,
        corr: ArrayLike,
        dt: float,
        T: int,
        sigma_factor: float = 1.0,
        rng=None,
    ) -> np.ndarray:
        """Returns a multivariate (N) geometric Brownian motion series.

        Args:
            Si (ArrayLike): Initial values of elements in the series, size N.
            mu (ArrayLike): Drift of the Brownian motion, size N.
            sigma (ArrayLike): Volatility of the Brownian motion, size N.
            corr (ArrayLike): Correlation matrix of the Brownian motion, size N x N.
            dt (float): Time step of the Brownian motion.
            T (int): Number of time steps in the series.
            rng (Optional[np.random.RandomState], optional): Random number generator.

        Returns:
            ArrayLike: Multivariate geometric Brownian motion series of size N x T.
        """
        Si = np.array(Si).flatten().astype(float)
        mu = np.array(mu).flatten().astype(float)
        sigma = np.array(sigma).flatten().astype(float)
        corr = np.array(corr).astype(float)

        assert len(Si) == len(mu), "Si and mu must have the same length"
        assert len(mu) == len(sigma), "mu and sigma must have the same length"
        assert len(mu) == len(corr), "mu and corr must have the same length"
        assert corr.shape[0] == corr.shape[1], "Correlation matrix must be square"
        assert len(corr.shape) == 2, "corr must be a 2D array"

        returns = self._multivariate_geometric_brownian_returns(
            mu, sigma, corr, dt, T - 1, sigma_factor, rng
        )
        values = np.cumprod(
            np.concatenate([Si.reshape(-1, 1), returns], axis=1), axis=1
        )

        return values

    def _generate_brownian_series_like(
        self,
        ts: ArrayLike,
        Si: Optional[ArrayLike] = None,
        num_elements: Optional[int] = None,
        zero_mu: bool = False,
        sigma_factor: float = 1.0,
        rng=None,
    ) -> np.ndarray:
        """Generate a Brownian motion series like the given time series.

        Args:
            ts (ArrayLike): Time series to generate the Brownian motion series like, (N, T).
            Si (Optional[ArrayLike], optional): Initial values of series, size N. Defaults
                ts[:, 0].
            num_elements (Optional[int], optional): Number of time steps in the series.
                Defaults to ts.shape[1].
            zero_mu (bool, optional): If True, the drift of the Brownian motion is set to 0.
                Defaults to False.
            rng (Optional[np.random.RandomState], optional): Random number generator.

        Returns:
            np.ndarray: Brownian motion series of size N x T.
        """
        ts = np.array(ts).astype(float)
        Si = ts[:, 0] if Si is None else np.array(Si).flatten().astype(float)
        num_elements = ts.shape[1] if num_elements is None else num_elements

        log_returns = np.log(ts[:, 1:] / ts[:, :-1])
        mu = log_returns.mean(axis=1) if not zero_mu else np.zeros(ts.shape[0])
        sigma = log_returns.std(axis=1)

        if log_returns.shape[0] == 1:
            corr = np.zeros((1, 1))
        else:
            corr = np.corrcoef(log_returns)

        brownian_series = self._multivariate_geometric_brownian_series(
            Si, mu, sigma, corr, 1.0, num_elements, sigma_factor, rng=rng
        )

        return brownian_series

    def _generate_brownian_continuation_from(
        self,
        ts: ArrayLike,
        num_elements: int,
        zero_mu: bool = False,
        sigma_factor: float = 1.0,
        rng=None,
    ) -> np.ndarray:
        ts = np.array(ts).astype(float)
        Si = ts[:, -1]
        brownian_series = self._generate_brownian_series_like(
            ts, Si, num_elements, zero_mu, sigma_factor, rng
        )

        return brownian_series[:, 1:]
