import pandas as pd

# We import the processing function from the specified file
from sustentable.core.data_shufler import data_shuffler  #


class TrainTestSliderController:
    """
    Object designed to integrate as a slider component in Pysual.
    It calculates the percentages and delegates the data processing to the backend.
    """

    def __init__(self, frame: pd.DataFrame, seed: int = 42, initial_test_pct: float = 20.0):
        # Validates that the input is a Pandas DataFrame[cite: 1]
        if type(frame) is not pd.DataFrame:
            raise TypeError("frame must be a Pandas Dataframe")  # [cite: 1]

        self.frame = frame  # [cite: 1]
        self.seed = seed  # [cite: 1]
        self.df_train = None
        self.df_test = None  # [cite: 1]

        # We perform the first split with the default values
        self.on_slider_move(initial_test_pct)

    def on_slider_move(self, test_slider_value: float) -> tuple[float, float]:
        """
        Callback method to connect with the slider's movement event.

        Parameters
        ----------
        test_slider_value : float
            The current slider value (0 to 100), representing the test %.

        Returns
        -------
        tuple[float, float]
            Updated percentages of (training, test) to reflect in the interface.
        """
        # We calculate the training percentage based on the test percentage
        train_slider_value = 100.0 - test_slider_value

        # We convert the value (0-100) to the fraction (0-1) required by data_shufler.py[cite: 1]
        test_percent = float(test_slider_value / 100.0)

        # We invoke the backend file's logic to perform the partition
        self.df_test, self.df_train = data_shuffler(
            frame=self.frame,  # [cite: 1]
            seed=self.seed,  # [cite: 1]
            test_percent=test_percent,  # [cite: 1]
        )

        # We return the percentages (0-100) so the Pysual interface can update its labels
        return train_slider_value, test_slider_value

    def get_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieves the subsets generated after adjusting the slider.
        """
        return self.df_train, self.df_test
