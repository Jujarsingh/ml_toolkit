import numpy as np
from typing import List, Dict

class Dataprocessor:
    """
    A class to handle data preprocessing tasks such as cleaning,
    outlier removal, normalization, and summary statistics.
    """
    def __init__(self, data: List[float]) -> None:
        """
        Initialize the DataProcessor with input data.

        Args:
             data(List[float]): Input dataset containing numerical values.
        """
        self.data: np.ndarray = np.array(data)

    def clean_missing(self) -> np.ndarray:
        """
        Remove Nan value from the dataset.

        Returns:
             np.ndarray: Cleaned dataset without missing value.
        """
        self.data = self.data[~np.isnan(self.data)]
        return self.data

    def remove_outliers(self) -> np.ndarray:
        """
        Remove outliers using mean ± 2*std deviation rule.

        Returns:
            np.ndarray: Dataset after outlier removal.
        """
        mean: float = float(np.mean(self.data))
        std:float = float(np.std(self.data))

        self.data = self.data[
            (self.data > mean - 2*std) & (self.data< mean + 2*std)
        ]
        return self.data

    def normalize(self) -> np.ndarray:
        """
        Normalize data using Min-Max scaling.

        Returns:
             np.ndarray: Normalized dataset.
        """
        self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return self.data

    def summary(self) -> Dict[str, float]:
        """
        Generate summary statistics of the dataset.
        """
        return {
            "mean": float(np.mean(self.data)),
            "std": float(np.std(self.data)),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data))
        }

    def __str__(self) -> str:
        """
        String representation of the Dataprocessor object.

        Returns:
             str: Description of dataset size.
        """
        return f"Data Processor with {len(self.data)} samples"