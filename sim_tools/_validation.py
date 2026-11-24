"""
Common validation methods for sim-tools functions and methods.

This module provides a composable validation framework for checking
input parameters for sim-tools code. Each validator follows the
Single Responsibility Principle, checking exactly one constraint, and can be
combined with others through the `validate()` function.

The module also provides validate_fit_data(), a standalone helper that
converts observed data to a NumPy array and checks common fitting
requirements. Unlike the composable validators, it returns the validated
data and is called directly.

Examples
--------
>>> from validation import validate, is_numeric, is_positive
>>>
>>> # Validate that a parameter is a positive number
>>> def calculate_square_root(x):
...     validate(x, "x", is_numeric, is_positive)
...     return math.sqrt(x)
>>>
>>> # Validate a probability parameter
>>> def bernoulli_trial(p):
...     validate(p, "p", is_numeric, is_probability)
...     return random.random() < p

Available Validators
-------------------
is_numeric : Checks if value is a number (int or float)
is_positive : Checks if value is positive (greater than zero)
is_probability : Checks if value is between 0 and 1 inclusive
is_integer: Check is the value is a whole number
is_non_negative: Checks if the value is >= 0.0
is_ordered_pair: Checks that low < high
is_ordered_triplet: Checks that low < middle < high

Fitting Data Validation
-----------------------
validate_fit_data : Converts data to a one-dimensional float64 array and
    checks that it contains at least two observations and no NaN or
    infinite values.

Notes
-----
Validation functions raise appropriate exceptions with descriptive messages:
- TypeError for type validation failures
- ValueError for value constraint failures
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Type for validator functions
ValidatorFunc = Callable[[Any, str], None]


def is_numeric(value: Any, name: str) -> None:
    """Validates that a value is a number (int or float)."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")


def is_positive(value: Any, name: str) -> None:
    """Validates that a value is positive (> 0)."""
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")


def is_non_negative(value: Any, name: str) -> None:
    """Validates that a value is greater than or equal to 0."""
    if not value >= 0:
        raise ValueError(f"{name} must be greater than or equal to 0, got {value}")


def is_probability(value: Any, name: str) -> None:
    """Validates that a value is a valid probability (between 0 and 1)."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def is_integer(value: Any, name: str) -> None:
    """Validates that a value is an integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


def validate(value: Any, name: str, *validators: ValidatorFunc) -> None:
    """Applies multiple validators to a value."""
    for validator in validators:
        validator(value, name)


def is_ordered_pair(
    low: Any, high: Any, low_name: str = "low", high_name: str = "high"
) -> None:
    """Validates that two values are in ascending order (low < high)."""
    if not low < high:
        raise ValueError(
            f"{low_name} must be less than {high_name}, got {low} >= {high}"
        )


def is_ordered_triplet(
    low: Any,
    middle: Any,
    high: Any,
    low_name: str = "low",
    middle_name: str = "middle",
    high_name: str = "high",
) -> None:
    """Validates that three values are in ascending order."""
    if not low < middle:
        raise ValueError(
            f"{low_name} must be less than {middle_name}, " + f"got {low} >= {middle}"
        )
    if not middle < high:
        raise ValueError(
            f"{middle_name} must be less than {high_name}, " + f"got {middle} >= {high}"
        )


def is_probability_vector(value: np.ndarray, name: str) -> None:
    """Validates that the array is a valid probability vector."""
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if np.any(value < 0) or np.any(value > 1):
        raise ValueError(f"All elements in {name} must be between 0 and 1")
    if not np.isclose(np.sum(value), 1.0, atol=1e-8):
        raise ValueError(f"{name} must sum to 1.0 (±1e-8 tolerance)")


def is_positive_array(value: np.ndarray, name: str) -> None:
    """Validates that all elements in the array are positive."""
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if np.any(value <= 0):
        raise ValueError(f"All elements in {name} must be positive (>0)")


def validate_fit_data(data: ArrayLike) -> NDArray[np.float64]:
    """Convert observed data to a float array and validate it for fitting.

    Parameters
    ----------
    data : ArrayLike
        Observed data to fit a distribution to.

    Returns
    -------
    numpy.ndarray
        A one-dimensional float64 array containing at least two finite
        observations.

    Raises
    ------
    TypeError
        If the input contains objects that cannot be converted to floats.
    ValueError
        If there are any validation issues for the data.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 1:
        raise ValueError("Data must be one-dimensional.")
    if data.size < 2:
        raise ValueError("Data must contain at least two observations.")
    if not np.isfinite(data).all():
        raise ValueError("Data must not contain NaN or infinite values.")

    return data
