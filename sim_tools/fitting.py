"""Fit sim-tools distributions to a data sample and compare them."""

import numpy as np
from numpy.random import SeedSequence
from numpy.typing import ArrayLike
from scipy.stats import ks_2samp
from statsmodels.graphics.gofplots import qqplot_2samples

from sim_tools.distributions import (
    Beta,
    Erlang,
    ErlangK,
    Exponential,
    Gamma,
    Lognormal,
    Normal,
    PearsonV,
    PearsonVI,
    Poisson,
    Triangular,
    Uniform,
    Weibull,
    RawContinuousEmpirical,
)

# Continuous distributions with a fit() method
DISTS = [
    Beta,
    Erlang,
    ErlangK,
    Exponential,
    Gamma,
    Lognormal,
    Normal,
    PearsonV,
    PearsonVI,
    Poisson,
    Triangular,
    Uniform,
    Weibull,
]

class FitDist:
    """Fit candidate distributions to one sample and rank them."""

    def __init__(self, data: ArrayLike, random_seed: int | SeedSequence | None = None):
        """TODO.

        Parameters
        ----------
        data : ArrayLike
            Observed data to fit distribution to.
        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError("Data must be one-dimensional.")
        if np.isnan(data).any():
            raise ValueError("Data contains NaN - drop or impute first.")

        self.data = data
        self.random_seed = random_seed
        self.results = {}

    def fit(self, dists=None):
        """Fit candidate distributions and calculate K-S statistic.

        Calculate the two-sample Kolmogorov-Smirnov (K-S) statistic by
        comparing the observed and generated samples. It ranges from 0 to 1,
        with smaller values indicating closer agreement between samples.

        Parameters
        ----------
        dists : list of type[Distribution], optional
            sim-tools distribution classes to fit, for example,
            `[Exponential, Gamma, Weibull]`. Pass classes, not instances.
            Each class must be included in `DISTS` and provide a `fit()`
            class method. If None, all classes in `DISTS` are used.

        Raises
        ------
        ValueError
            If any requested class is not in DISTS.
        """
        if dists is None:
            dists = DISTS

        unsupported = [d for d in dists if d not in DISTS]
        if unsupported:
            requested = ", ".join(getattr(d, "__name__", repr(d)) for d in unsupported)
            available = ", ".join(d.__name__ for d in DISTS)
            raise ValueError(
                f"Cannot fit: {requested}. Supported distributions: {available}."
            )

        # Fit distribution, generate sample, and calculate Kolmogorov-Smirnov
        for dist in dists:
            try:
                fitted = dist.fit(self.data, random_seed=self.random_seed)
                sample = fitted.sample(size=max(len(self.data), 10_000))
                ks_statistic = ks_2samp(self.data, sample).statistic
                self.results[dist.__name__] = {
                    "distribution": fitted,
                    "params": vars(fitted),
                    "ks_statistic": ks_statistic,
                    "error": None
                }
            except ValueError as err:
                self.results[dist.__name__] = {
                    "distribution": None,
                    "params": None,
                    "ks_statistic": np.nan,
                    "error": str(err)
                }
