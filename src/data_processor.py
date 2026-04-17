
import numpy as np
from typing import List, Dict
from src.logger import get_logger
from src.exceptions import DataError

logger = get_logger(__name__)

class DataProcessor:
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
        if not isinstance(data, list):
            logger.error("Input data is not a list")
            raise DataError("Input data must be a list")

        if len(data) == 0:
            logger.error("Empty dataset provided")
            raise DataError("Data cannot be empty")

        try:
            self.data: np.ndarray = np.array(data, dtype=float)
            logger.info("Data successfully converted to numpy array")
        except Exception as e:
            logger.error(f"Data conversion failed: {e}")
            raise DataError("Invalid data format")


    def clean_missing(self) -> np.ndarray:
        """
        Remove Nan value from the dataset.

        Returns:
             np.ndarray: Cleaned dataset without missing value.
        """
        initial_size = len(self.data)

        self.data = self.data[~np.isnan(self.data)]

        removed = initial_size - len(self.data)
        logger.info(f"Removed {removed} missing values")

        return self.data

    def remove_outliers(self) -> np.ndarray:
        """
        Remove outliers using mean ± 2*std deviation rule.

        Returns:
            np.ndarray: Dataset after outlier removal.
        """
        mean: float = float(np.mean(self.data))
        std:float = float(np.std(self.data))

        initial_size = len(self.data)

        self.data = self.data[
            (self.data > mean - 2*std) & (self.data < mean + 2 * std)
        ]

        removed = initial_size - len(self.data)
        logger.info(f"Removed {removed} outliers")

        return self.data

    def normalize(self) -> np.ndarray:
        """
        Normalize data using Min-Max scaling.

        Returns:
             np.ndarray: Normalized dataset.
        """

        min_val = np.min(self.data)
        max_val = np.max(self.data)

        if min_val == max_val:
            logger.error("Normalization failed: all values are identical")
            raise DataError("Cannot normalize when all values are the same")

        self.data = (self.data - min_val) / (max_val - min_val)

        logger.info("Data normalization completed")
        return self.data

    def summary(self) -> Dict[str, float]:
        """
        Generate summary statistics of the dataset.

        Returns:
            Dict[str, float]: Dictionary containing mean, std, min, max.
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