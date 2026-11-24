"""Fit sim-tools distributions to a data sample and compare them."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from numpy.typing import ArrayLike
from scipy.stats import ks_2samp
from statsmodels.graphics.gofplots import qqplot_2samples

from sim_tools._validation import (
    is_integer,
    is_positive,
    validate,
    validate_fit_data,
)
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
)

# Candidate distributions with a fit() methodfig
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
    """Fit candidate distributions to one sample and compare their fit."""

    def __init__(self, data: ArrayLike, random_seed: int | SeedSequence | None = None):
        """Initialise distribution fitter with observed data.

        Parameters
        ----------
        data : ArrayLike
            Observed data to fit distribution to.
        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        self.data = validate_fit_data(data)
        self.random_seed = random_seed
        self.results = {}

    def fit(self, dists=None, verbose: bool = True) -> None:
        """Fit candidate distributions and calculate K-S statistic.

        Calculate the two-sample Kolmogorov-Smirnov (K-S) statistic by
        comparing the observed and generated samples. It ranges from 0 to 1,
        with smaller values indicating closer agreement between samples.
        Because the comparison uses a generated sample, results can vary with
        the random seed.

        Parameters
        ----------
        dists : list of type[Distribution], optional
            sim-tools distribution classes to fit, for example,
            `[Exponential, Gamma, Weibull]`. Pass classes, not instances.
            Each class must be included in `DISTS` and provide a `fit()`
            class method. If None, all classes in `DISTS` are used.
        verbose : bool, default=True
            Print the ranked results and guidance on interpreting and
            using them. Set to False to suppress printing.

        Raises
        ------
        ValueError
            If any requested class is not in DISTS.
        """
        self.results = {}

        if dists is None:
            dists = DISTS

        unsupported = [d for d in dists if d not in DISTS]
        if unsupported:
            requested = ", ".join(getattr(d, "__name__", repr(d)) for d in unsupported)
            available = ", ".join(d.__name__ for d in DISTS)
            raise ValueError(
                f"Cannot fit: {requested}. Supported distributions: {available}."
            )

        for dist in dists:
            try:
                # Fit distribution and generate the comparison sample
                # Generate at least 10,000 observations, or match the data
                # size if larger
                fitted = dist.fit(self.data, random_seed=self.random_seed)
                sample = fitted.sample(size=max(len(self.data), 10_000))

                # Calculate Kolmogorov-Smirnov
                ks_statistic = ks_2samp(self.data, sample).statistic

                # Save results
                self.results[dist.__name__] = {
                    "params": fitted.get_params(),
                    "ks_statistic": ks_statistic,
                    "sample": sample,
                }

            # If there's an error, save the message
            except ValueError as err:
                self.results[dist.__name__] = {"error": str(err)}

        if verbose:
            self.summary()

    def summary(self) -> None:
        """Print ranked fitting results and explain their interpretation."""
        if not self.results:
            print("No fitting results available. Call fit() first.")
            return

        table = self.get_results()

        print("Distribution fitting results")
        print("============================")
        print(
            "\nThe two-sample Kolmogorov-Smirnov (K-S) statistic compares the\n"
            "observed and generated samples. It ranges from 0 to 1, with smaller\n"
            "values indicating closer agreement between samples. Results are\n"
            "ranked from smallest to largest K-S statistic.\n"
        )
        print(
            table.drop(columns="params").to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
                na_rep="-",
            )
        )

        print(
            "\nNext steps"
            "\n----------"
            "\nDo not choose a distribution based on the K-S statistic alone.\n"
            "Use plot() to inspect Q-Q plots and histograms before deciding\n"
            "which distribution is suitable for your data. For example:\n"
            "\n  plot(dist=Gamma)  Plot a chosen distribution.\n"
            "  plot(top=3)       Plot the top three successful fits.\n"
            "\nUsing your distribution"
            "\n-----------------------"
            "\nYou can find the fitted parameters in the 'params' column of the\n"
            "DataFrame returned by get_results(), or access them directly from\n"
            "your FitDist instance. For example:\n"
            "\n  fitter = FitDist(data)\n"
            "  fitter.fit()\n"
            "  params = fitter.results['Gamma']['params']\n"
            "  dist = Gamma(**params)"
        )

    def get_results(self) -> pd.DataFrame:
        """Return fitting results ranked by K-S statistic.

        Returns
        -------
        pandas.DataFrame
            Distribution names, K-S statistics, constructor parameters,
            and fitting errors. Failed fits appear last. Generated
            samples are excluded.
        """
        results = (
            pd.DataFrame.from_dict(
                self.results,
                orient="index",
                columns=["ks_statistic", "params", "error"],
            )
            .rename_axis("distribution")
            .reset_index()
            .sort_values(
                "ks_statistic",
                na_position="last",
                kind="stable",
                ignore_index=True,
            )
        )

        if results["error"].isna().all():
            results = results.drop(columns="error")

        return results

    def plot(self, dist=None, *, top=None):
        """Create Q-Q plots and histograms for selected fitted distributions.

        If neither dist nor top is specified, plot all successful fits in
        ascending order of K-S statistic. Reuse the comparison samples
        generated by fit(); do not refit distributions or generate new
        samples.

        Parameters
        ----------
        dist : type[Distribution], optional
            Distribution class to plot, for example, Gamma. Cannot be
            supplied together with top.
        top : int, optional
            Number of successful candidates to plot, ranked by ascending
            K-S statistic. If larger than the number of successful fits,
            all successful fits are plotted.

        Returns
        -------
        dict
            Dictionary keyed by distribution name. Each entry contains
            the Matplotlib figures under 'qq_plot' and 'hist_plot'.
        """
        if not self.results:
            raise ValueError("Call fit() before requesting plots.")

        if dist is not None and top is not None:
            raise ValueError("Provide either dist or top, not both.")

        # If running for a specified distribution
        if dist is not None:
            if dist not in DISTS:
                raise ValueError("dist must be a supported distribution class.")

            dist_name = dist.__name__
            result = self.results.get(dist_name)

            if result is None:
                raise ValueError(f"{dist_name} has not been fitted.")

            if "error" in result:
                raise ValueError(f"{dist_name} could not be fitted: {result['error']}")

            selected = [dist]

        # If running for the top X distributions
        else:
            if top is not None:
                validate(top, "top", is_integer, is_positive)

            ranked = self.get_results().dropna(subset=["ks_statistic"])
            names = ranked["distribution"].tolist()

            if not names:
                raise ValueError(
                    "No successful fits are available to plot. "
                    "See summary() for fitting errors."
                )

            # If specified e.g., top=3, filter to the top 3
            if top is not None:
                names = names[:top]

            # For each dist in DISTS, pair its name with the class itself
            classes = {candidate.__name__: candidate for candidate in DISTS}
            # Look up the selected class
            selected = [classes[name] for name in names]

        return {
            candidate.__name__: {
                "qq_plot": self.qq_plot(candidate),
                "hist_plot": self.hist_plot(candidate),
            }
            for candidate in selected
        }

    def qq_plot(self, dist):
        """Create a Q-Q plot for a fitted distribution.

        Compare quantiles of the observed sample on the horizontal axis
        with quantiles of the generated sample on the vertical axis.
        Include a 45-degree reference line.

        Parameters
        ----------
        dist : distribution class
            Distribution class to plot, for example, Gamma. Pass a class,
            not an instance or name. Must have been fitted successfully.

        Returns
        -------
        matplotlib.figure.Figure
            Q-Q plot comparing the observed and generated samples.
        """
        qq_fig = qqplot_2samples(
            data1=self.data,
            data2=self.results[dist.__name__]["sample"],
            xlabel="Observed",
            ylabel="Fitted",
            line="45",
        )
        qq_fig.suptitle(dist.__name__)
        return qq_fig

    def hist_plot(self, dist):
        """Create overlaid histograms of observed and generated samples.

        Plot density-normalised histograms using 200 shared bins calculated
        from the combined observed and generated samples.

        Parameters
        ----------
        dist : distribution class
            Distribution class to plot, for example, Gamma. Pass a class,
            not an instance or name. Must have been fitted successfully.

        Returns
        -------
        matplotlib.figure.Figure
            Overlaid histograms of the observed and generated samples.
        """
        sample = self.results[dist.__name__]["sample"]

        bins = np.histogram_bin_edges(
            np.concatenate([self.data, sample]),
            bins=200,
        )
        hist_fig, ax = plt.subplots()
        ax.hist(
            self.data,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Observed",
        )
        ax.hist(
            sample,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Fitted",
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(dist.__name__)
        ax.legend()
        hist_fig.tight_layout()
        return hist_fig
