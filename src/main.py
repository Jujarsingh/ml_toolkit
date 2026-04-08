
from typing import List
from data_processor import Dataprocessor
from model import SimpleModel
import numpy as np


def main() -> None:
    """
    Main function to run data processing and model workflow.
    """
    data: List[float] = [1, 2, 3, 4, 100, np.nan, 5]

    processor = Dataprocessor(data)
    processor.clean_missing()
    processor.remove_outliers()
    processor.normalize()

    print("Summary:", processor.summary())

    model = SimpleModel()
    model.train(processor.data.tolist())

    predictions = model.predict([1, 2, 3])
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()

