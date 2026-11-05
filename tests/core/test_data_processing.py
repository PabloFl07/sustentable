"""Unit tests for missing-value detection and each treatment strategy."""

import math

import pytest
from pandas import DataFrame

from sustentable.core.data_processing import NanStrategy, detect_nan_values, has_nan_values, parse_constant, process_nan_values

NAN = math.nan
COLUMNS = ["x", "y"]


@pytest.fixture
def train() -> DataFrame:
    """Training rows with one missing x and one missing y; ``note`` is not selected."""
    return DataFrame({"x": [1.0, NAN, 3.0, 8.0], "y": [10.0, 20.0, NAN, 40.0], "note": [None, "a", "b", "c"]})


@pytest.fixture
def test() -> DataFrame:
    """Test rows with one missing x."""
    return DataFrame({"x": [NAN, 5.0], "y": [50.0, 60.0], "note": ["d", None]})


def test_detect_counts_each_selected_column_per_subset(train: DataFrame, test: DataFrame) -> None:
    """Counts are reported per column and subset, ignoring unselected columns and repeated labels."""
    report = detect_nan_values(train, test, ["x", "y", "x"])
    assert report.to_dict("records") == [{"column": "x", "train": 1, "test": 1}, {"column": "y", "train": 1, "test": 0}]
    assert has_nan_values(train, test, COLUMNS)


def test_detect_without_missing_values() -> None:
    """Clean data is reported as such, even when unselected columns have missing values."""
    clean = DataFrame({"x": [1, 2], "other": [NAN, NAN]})
    assert detect_nan_values(clean, clean, ["x"]).train.sum() == 0
    assert not has_nan_values(clean, clean, ["x"])


@pytest.mark.parametrize("columns", [[], ["missing"]])
def test_detect_rejects_invalid_columns(train: DataFrame, test: DataFrame, columns: list[str]) -> None:
    """An empty selection or an unknown column is an error."""
    with pytest.raises(ValueError):
        detect_nan_values(train, test, columns)


def test_delete_removes_affected_rows_in_both_subsets(train: DataFrame, test: DataFrame) -> None:
    """Only missing values in the selected columns cause a row to be removed."""
    result = process_nan_values(train, test, COLUMNS, NanStrategy.DELETE)
    assert result.train.x.tolist() == [1.0, 8.0] and result.test.x.tolist() == [5.0]
    assert (result.train_changes, result.test_changes) == (2, 1)
    assert list(result.train.index) == [0, 1]


def test_delete_refuses_to_empty_a_subset(train: DataFrame) -> None:
    """Deleting must leave rows in both subsets."""
    all_missing = DataFrame({"x": [NAN], "y": [1.0]})
    with pytest.raises(ValueError, match="test subset empty"):
        process_nan_values(train, all_missing, COLUMNS, NanStrategy.DELETE)


@pytest.mark.parametrize(("strategy", "x_fill", "y_fill"), [(NanStrategy.MEAN, 4.0, 70 / 3), (NanStrategy.MEDIAN, 3.0, 20.0)])
def test_mean_and_median_use_training_values_for_both_subsets(
    train: DataFrame, test: DataFrame, strategy: NanStrategy, x_fill: float, y_fill: float
) -> None:
    """Fill values come from the training subset only and are reused for the test subset."""
    result = process_nan_values(train, test, COLUMNS, strategy)
    assert result.train.x[1] == x_fill and result.test.x[0] == x_fill
    assert result.train.y[2] == pytest.approx(y_fill)
    assert (result.train_changes, result.test_changes) == (2, 1)
    assert result.train.note.isna().sum() == 1, "Unselected columns are left as they are."


def test_constant_fills_selected_columns(train: DataFrame, test: DataFrame) -> None:
    """The constant replaces every missing value in the selected columns."""
    result = process_nan_values(train, test, COLUMNS, "constant", -1.5)
    assert result.train.x[1] == -1.5 and result.train.y[2] == -1.5 and result.test.x[0] == -1.5
    assert not has_nan_values(result.train, result.test, COLUMNS)


def test_inputs_are_not_modified(train: DataFrame, test: DataFrame) -> None:
    """Every strategy works on copies."""
    before = train.copy(), test.copy()
    for strategy in NanStrategy:
        process_nan_values(train, test, COLUMNS, strategy, 0.0)
    assert train.equals(before[0]) and test.equals(before[1])


@pytest.mark.parametrize("value", [None, "1", math.inf, True])
def test_constant_requires_a_finite_number(train: DataFrame, test: DataFrame, value: object) -> None:
    """The constant strategy rejects a missing or non-numeric constant."""
    with pytest.raises(ValueError):
        process_nan_values(train, test, COLUMNS, NanStrategy.CONSTANT, value)


@pytest.mark.parametrize("strategy", [NanStrategy.MEAN, NanStrategy.MEDIAN])
def test_mean_and_median_need_numeric_training_values(test: DataFrame, strategy: NanStrategy) -> None:
    """Text columns and all-missing training columns cannot produce a fill value."""
    text = DataFrame({"x": ["a", None], "y": [NAN, NAN]})
    with pytest.raises(ValueError, match="not numeric"):
        process_nan_values(text, test, ["x"], strategy)
    with pytest.raises(ValueError, match="no values"):
        process_nan_values(text, test, ["y"], strategy)


def test_unknown_strategy(train: DataFrame, test: DataFrame) -> None:
    """Only the listed strategies are accepted."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        process_nan_values(train, test, COLUMNS, "mode")


@pytest.mark.parametrize(("text", "expected"), [("2", 2.0), (" -1.5 ", -1.5), ("3,25", 3.25), ("1e3", 1000.0)])
def test_parse_constant_accepts_numbers(text: str, expected: float) -> None:
    """Integers, decimals with either separator and scientific notation are accepted."""
    assert parse_constant(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "nan", "inf", "1.2.3"])
def test_parse_constant_rejects_other_text(text: str) -> None:
    """Empty, non-numeric and non-finite text is rejected."""
    with pytest.raises(ValueError):
        parse_constant(text)
