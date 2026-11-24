"""Tests for FitDist."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from sim_tools.distributions import (
    Beta,
    Exponential,
    Gamma,
    Poisson,
    RawContinuousEmpirical,
)
from sim_tools.fitting import FitDist


@pytest.fixture
def fitter():
    """Fit two candidates to a reproducible positive sample."""
    data = np.random.default_rng(42).exponential(scale=5, size=500)

    fitter = FitDist(data, random_seed=42)
    fitter.fit(dists=[Exponential, Gamma], verbose=False)

    return fitter


def test_fitdist_results_are_ranked(fitter):
    """Return successful candidates ranked by ascending K-S statistic."""
    results = fitter.get_results()

    assert set(results["distribution"]) == {"Exponential", "Gamma"}
    assert results["ks_statistic"].between(0, 1).all()
    assert results["ks_statistic"].is_monotonic_increasing
    assert "error" not in results.columns

    for name in results["distribution"]:
        assert isinstance(fitter.results[name]["params"], dict)
        assert len(fitter.results[name]["sample"]) == 10_000


def test_fitdist_plot_top_candidate(fitter):
    """Return both plot types for the highest-ranked candidate."""
    expected_name = fitter.get_results().iloc[0]["distribution"]

    figures = fitter.plot(top=1, show=False)

    assert list(figures) == [expected_name]
    assert set(figures[expected_name]) == {"qq_plot", "hist_plot"}
    assert all(isinstance(fig, Figure) for fig in figures[expected_name].values())


def test_fitdist_accepts_supported_distributions():
    """Accept a list containing only supported distribution classes."""
    data = np.random.default_rng(42).poisson(lam=5, size=500)
    fitter = FitDist(data, random_seed=42)

    fitter.fit(dists=[Beta, Poisson], verbose=False)

    assert set(fitter.results) == {"Beta", "Poisson"}


def test_fitdist_rejects_unsupported_distributions():
    """Reject a list containing an unsupported distribution class."""
    data = np.random.default_rng(42).poisson(lam=5, size=500)
    fitter = FitDist(data, random_seed=42)

    with pytest.raises(ValueError, match="Cannot fit: RawContinuousEmpirical"):
        fitter.fit(
            dists=[Beta, Poisson, RawContinuousEmpirical],
            verbose=False,
        )
