
from typing import List
import numpy as np

from src.data_processor import DataProcessor
from src.model import SimpleModel
from src.exceptions import DataError
from src.logger import get_logger

logger = get_logger(__name__)



def main() -> None:
    """
    Main function to run data processing and model workflow.
    """

    logger.info("Starting Ml pipeline")

    try:
        # Input data
        data: List[float] = [1, 2, 3, 4, 100, np.nan, 5]

        # Data processing
        processor = DataProcessor(data)

        logger.info("Cleaning missing data")
        processor.clean_missing()

        logger.info("Removing outliers")
        processor.remove_outliers()

        logger.info("Normalizing data")
        processor.normalize()

        summary = processor.summary()
        print("Summary:", summary)

        # Model
        model = SimpleModel()

        logger.info("Training model")
        model.train(processor.data.tolist())

        logger.info("Making prediction")
        predictions = model.predict([1, 2, 3])

        print("Predictions:", predictions)

        logger.info("Pipeline completed successfully")

    except DataError as e:
        logger.error(f"Data error occurred:{e}")
        print(f"Data Error: {e}")

    except Exception as e :
        logger.critical(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

