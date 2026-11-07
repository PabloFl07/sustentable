"""Split a dataset into training and test subsets with scikit-learn."""

import math

from pandas import DataFrame
from sklearn.model_selection import train_test_split


def count_test_rows(total_rows: int, test_fraction: float) -> int:
    """Return how many of ``total_rows`` ``split_dataset`` puts in the test subset; scikit-learn rounds up."""
    return math.ceil(test_fraction * total_rows)


def split_dataset(frame: DataFrame, test_fraction: float, seed: int) -> tuple[DataFrame, DataFrame]:
    """Shuffle the rows of ``frame`` and split them into training and test subsets.

    Args:
        frame: Dataset to split. It is not modified.
        test_fraction: Share of rows assigned to the test subset, strictly between 0 and 1.
        seed: Random seed, so the same inputs always produce the same subsets.

    Returns:
        The training and test subsets, each with a fresh ``0..n-1`` index.

    Raises:
        ValueError: If ``test_fraction`` is not strictly between 0 and 1 or either subset would be empty.
    """
    train, test = train_test_split(frame, test_size=test_fraction, random_state=seed, shuffle=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)
