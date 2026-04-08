from typing import List


class SimpleModel:
    """
    A simple mock model class to simulate training,
    prediction, and evaluation processes.
    """

    def __init__(self) -> None:
        """
        Initialize the model state.
        """
        self.is_trained = False

    def train(self, data: List[float]) -> None:
        """
        Simulate model training.

        Args:
            data (List[float]): Training dataset.
        """
        print("Training model...")
        self.is_trained = True

    def predict(self, x: List[float]) -> List[float]:
        """
        Simulate predictions on input data.

        Args:
             x (List[float]): Input values.

        Returns:
            List[float]: Predicted values.

        Raises:
            Exception; If model is not trained.
        """
        if not self.is_trained:
            raise Exception("Model not trained")
        return [i * 2 for i in x]

    def evaluate(self) -> str:
        """
        Simulate model evaluation.

        Returns:
             str: Evaluation result message.
        """
        return "Model evaluation complete"