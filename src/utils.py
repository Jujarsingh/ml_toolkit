"""
Utility functions for basic data processing.
"""

import numpy as np



def calculate_mean(data):
    """
    Calculate mean of a list of numbers.

    :param data: List of Values (must be a non-empty list).
    :return: float - mean(average) of the list.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input must be a non-empty list")

    return float(np.mean(data))


def calculate_variance(data):
    """
    Calculate variance of a list of numbers.

    :param data: List of Values (must be a non-empty list).
    :return: float - variance of the list.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input must be a non-empty list")

    return float(np.var(data))


def normalize_data(data):
    """
    Normalize data using Min-Max scaling.

    :param data: List of Values (must be a non-empty list).
    :return: List of normalized data using Min-Max scaling.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input must be a non-empty list")

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        raise ValueError("All values are the same, cannot normalize.")

    normalized = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized