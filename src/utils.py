"""
Utility functions for basic data processing.
"""

import numpy as np



def calculate_mean(data: list[float]) -> float:
    """
    Calculate mean of a list of numbers.

    Args:
        data (list[float]): Input list of numbers.

    Returns:
        float: Mean value.

    Raises:
        ValueError: If list is empty.
        Type error: If input is not list.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list.")

    if len(data) == 0:
        raise ValueError("List cannot be empty.")

    return float(np.mean(data))


def calculate_variance(data: list[float]) -> float:
    """
    Calculate variance of a list of numbers.

    Args:
        data (list[float]): Input list of numbers.

    Returns:
        float: Variance of list of numbers.

    Raises:
        ValueError: If list is empty.
        Type error: If input is not list.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list.")

    if len(data) == 0:
        raise ValueError("List cannot be empty.")

    return float(np.var(data))


def normalize_data(data: list[float]) -> list[float]:
    """
    Normalize data using Min-Max scaling.

    Args:
        data (list[float]): Input list of numbers.

    Returns:
        list: list of normalized data.

    Raises:
        ValueError: If list is empty or All values are the same.
        Type error: If input is not list.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")

    if len(data) == 0:
        raise ValueError("List cannot be empty.")

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        raise ValueError("All values are the same, cannot normalize.")

    normalized = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized