"""Detect and treat missing values in the selected columns of the train and test subsets.

Every function works on copies: the subsets passed in, and the file on disk, are never changed.
Fill values for the mean and median strategies come from the training subset only and are reused for the test subset.
"""

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from pandas import DataFrame
from pandas.api.types import is_numeric_dtype


class NanStrategy(StrEnum):
    """Available ways to treat missing values."""

    DELETE = "delete"
    MEAN = "mean"
    MEDIAN = "median"
    CONSTANT = "constant"


@dataclass(frozen=True)
class PreprocessResult:
    """Processed subsets and how many rows were removed or values were filled in each one."""

    train: DataFrame
    test: DataFrame
    strategy: NanStrategy
    train_changes: int
    test_changes: int


def detect_nan_values(train: DataFrame, test: DataFrame, columns: Iterable[Hashable]) -> DataFrame:
    """Count the missing values of each selected column, separately for the train and test subsets.

    Args:
        train: Training subset.
        test: Test subset.
        columns: Selected input and output columns. Repeated labels are counted once.

    Returns:
        A frame with one row per column and the ``column``, ``train`` and ``test`` columns.

    Raises:
        ValueError: If no column is selected or a column is missing from either subset.
    """
    selected = _selected_columns(train, test, columns)
    return DataFrame(
        {
            "column": selected,
            "train": [int(train[column].isna().sum()) for column in selected],
            "test": [int(test[column].isna().sum()) for column in selected],
        }
    )


def has_nan_values(train: DataFrame, test: DataFrame, columns: Iterable[Hashable]) -> bool:
    """Return whether any selected column has a missing value in either subset."""
    selected = _selected_columns(train, test, columns)
    return bool(train[selected].isna().to_numpy().any() or test[selected].isna().to_numpy().any())


def parse_constant(text: str) -> float:
    """Convert the user's constant to a finite number.

    Both ``.`` and ``,`` are accepted as the decimal separator.

    Raises:
        ValueError: If the text is empty, not numeric, or not finite.
    """
    cleaned = text.strip().replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"The constant value must be a number, got {text.strip()!r}.") from None
    if not math.isfinite(value):
        raise ValueError("The constant value must be a finite number.")
    return value


def process_nan_values(
    train: DataFrame,
    test: DataFrame,
    columns: Iterable[Hashable],
    strategy: NanStrategy | str,
    constant_value: float | None = None,
) -> PreprocessResult:
    """Apply one strategy to the missing values of the selected columns in both subsets.

    Args:
        train: Training subset.
        test: Test subset.
        columns: Selected input and output columns.
        strategy: How to treat missing values.
        constant_value: Number used by the constant strategy.

    Returns:
        The processed copies and, per subset, the rows removed (delete) or values filled (other strategies).

    Raises:
        ValueError: If the strategy is unknown, the constant is missing or not finite, a column cannot be
            averaged, the training subset has no values to compute a fill value, or deleting rows would
            leave a subset empty.
    """
    try:
        strategy = NanStrategy(strategy)
    except ValueError:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose one of: {', '.join(NanStrategy)}.") from None
    selected = _selected_columns(train, test, columns)

    if strategy is NanStrategy.DELETE:
        processed_train = train.dropna(subset=selected).reset_index(drop=True)
        processed_test = test.dropna(subset=selected).reset_index(drop=True)
        if processed_train.empty or processed_test.empty:
            emptied = "training" if processed_train.empty else "test"
            raise ValueError(f"Deleting the rows with missing values would leave the {emptied} subset empty.")
        return PreprocessResult(processed_train, processed_test, strategy, len(train) - len(processed_train), len(test) - len(processed_test))

    fill_values = _fill_values(train, selected, strategy, constant_value)
    train_changes = int(train[selected].isna().to_numpy().sum())
    test_changes = int(test[selected].isna().to_numpy().sum())
    return PreprocessResult(train.fillna(fill_values), test.fillna(fill_values), strategy, train_changes, test_changes)


def _fill_values(train: DataFrame, columns: list[Hashable], strategy: NanStrategy, constant_value: float | None) -> dict[Hashable, float]:
    """Compute one fill value per column; mean and median use the training subset only."""
    if strategy is NanStrategy.CONSTANT:
        if constant_value is None:
            raise ValueError("A constant value must be provided.")
        if isinstance(constant_value, bool) or not isinstance(constant_value, int | float) or not math.isfinite(constant_value):
            raise ValueError("The constant value must be a finite number.")
        return dict.fromkeys(columns, float(constant_value))

    values = {}
    for column in columns:
        series = train[column]
        if not is_numeric_dtype(series):
            raise ValueError(f"Column '{column}' is not numeric, so its {strategy} cannot be computed.")
        value = series.mean() if strategy is NanStrategy.MEAN else series.median()
        if math.isnan(value):
            raise ValueError(f"Column '{column}' has no values in the training subset to compute its {strategy}.")
        values[column] = float(value)
    return values


def _selected_columns(train: DataFrame, test: DataFrame, columns: Iterable[Hashable]) -> list[Hashable]:
    """Remove repeated labels and check that every column exists in both subsets."""
    selected = list(dict.fromkeys(columns))
    if not selected:
        raise ValueError("At least one column must be selected.")
    for column in selected:
        if column not in train.columns or column not in test.columns:
            raise ValueError(f"Column '{column}' is not available in the dataset.")
    return selected
