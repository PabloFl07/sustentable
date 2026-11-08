"""Unit tests for missing-value detection and each treatment strategy."""

import math

import pandas as pd
import pytest
from pandas import DataFrame

from sustentable.core.processing.data_processing import NanStrategy, detect_nan_values, has_nan_values, parse_constant, process_nan_values

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


# ----------------------------------------------------------------------------
# Nullable integer columns (Int64 with pd.NA)
# ----------------------------------------------------------------------------


@pytest.fixture
def int_train() -> DataFrame:
    """Training rows with a nullable integer column; the mean (4) and median (3) of ``n`` are whole numbers."""
    return DataFrame({"n": pd.array([1, pd.NA, 3, 8], dtype="Int64")})


@pytest.fixture
def int_test() -> DataFrame:
    """Test rows with a nullable integer column."""
    return DataFrame({"n": pd.array([pd.NA, 5], dtype="Int64")})


def test_detect_counts_pd_na_in_int64(int_train: DataFrame, int_test: DataFrame) -> None:
    """``pd.NA`` in an ``Int64`` column counts as a missing value."""
    assert detect_nan_values(int_train, int_test, ["n"]).to_dict("records") == [{"column": "n", "train": 1, "test": 1}]
    assert has_nan_values(int_train, int_test, ["n"])


@pytest.mark.parametrize(
    ("strategy", "constant", "train_n", "test_n"),
    [
        (NanStrategy.DELETE, None, [1, 3, 8], [5]),
        (NanStrategy.MEAN, None, [1, 4, 3, 8], [4, 5]),
        (NanStrategy.MEDIAN, None, [1, 3, 3, 8], [3, 5]),
        (NanStrategy.CONSTANT, 0.0, [1, 0, 3, 8], [0, 5]),
    ],
)
def test_int64_with_whole_fill_value_stays_int64(
    int_train: DataFrame, int_test: DataFrame, strategy: NanStrategy, constant: float | None, train_n: list[int], test_n: list[int]
) -> None:
    """When the fill value is a whole number, every strategy removes the ``pd.NA`` and keeps the ``Int64`` type."""
    result = process_nan_values(int_train, int_test, ["n"], strategy, constant)
    assert result.train.n.tolist() == train_n and result.test.n.tolist() == test_n
    assert result.train.n.dtype == result.test.n.dtype == "Int64"


@pytest.mark.xfail(strict=True, reason="pandas raises TypeError when a non-whole fill value is put into an Int64 column")
@pytest.mark.parametrize(("strategy", "constant"), [(NanStrategy.MEAN, None), (NanStrategy.MEDIAN, None), (NanStrategy.CONSTANT, 2.5)])
def test_int64_with_fractional_fill_value_does_not_leak_type_error(strategy: NanStrategy, constant: float | None) -> None:
    """A fill value with decimals in an ``Int64`` column either works or raises ``ValueError``, never a pandas error."""
    # Mean 4.25 and median 3.5, so no strategy has a whole fill value.
    train = DataFrame({"n": pd.array([1, pd.NA, 2, 9, 5], dtype="Int64")})
    test = DataFrame({"n": pd.array([pd.NA, 5], dtype="Int64")})
    try:
        process_nan_values(train, test, ["n"], strategy, constant)
    except ValueError:
        pass


# ----------------------------------------------------------------------------
# Unselected columns and column types
# ----------------------------------------------------------------------------


@pytest.fixture
def mixed_train() -> DataFrame:
    """Training rows with one column of each common type. Only ``x`` is selected; the others also have missing values."""
    return DataFrame(
        {
            "x": [1.0, NAN, 3.0, 8.0],
            "count": [1, 2, 3, 4],
            "nullable": pd.array([pd.NA, 2, 3, 4], dtype="Int64"),
            "ratio": [0.5, NAN, 1.5, 2.5],
            "label": pd.array(["a", "b", None, "d"], dtype="str"),
            "flag": [True, False, True, False],
        }
    )


@pytest.fixture
def mixed_test() -> DataFrame:
    """Test rows with the same columns as ``mixed_train``."""
    return DataFrame(
        {
            "x": [NAN, 5.0],
            "count": [5, 6],
            "nullable": pd.array([5, pd.NA], dtype="Int64"),
            "ratio": [NAN, 3.5],
            "label": pd.array([None, "f"], dtype="str"),
            "flag": [True, True],
        }
    )


@pytest.mark.parametrize("strategy", list(NanStrategy))
def test_unselected_columns_are_left_intact(mixed_train: DataFrame, mixed_test: DataFrame, strategy: NanStrategy) -> None:
    """Every strategy leaves the unselected columns as they were, missing values included, in the rows it keeps."""
    result = process_nan_values(mixed_train, mixed_test, ["x"], strategy, 0.0)
    for original, processed in [(mixed_train, result.train), (mixed_test, result.test)]:
        # DELETE drops the rows where x is missing; the other strategies keep every row.
        kept = original[original.x.notna()] if strategy is NanStrategy.DELETE else original
        expected = kept.drop(columns="x").reset_index(drop=True)
        pd.testing.assert_frame_equal(processed.drop(columns="x"), expected)


@pytest.mark.parametrize("strategy", list(NanStrategy))
def test_column_types_are_kept(mixed_train: DataFrame, mixed_test: DataFrame, strategy: NanStrategy) -> None:
    """No strategy changes the type of any column, selected or not."""
    result = process_nan_values(mixed_train, mixed_test, ["x", "ratio"], strategy, 0.0)
    pd.testing.assert_series_equal(result.train.dtypes, mixed_train.dtypes)
    pd.testing.assert_series_equal(result.test.dtypes, mixed_test.dtypes)


# ----------------------------------------------------------------------------
# DELETE limits and CONSTANT on text
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("train_x", "test_x", "emptied"),
    [
        ([NAN, NAN], [1.0], "training"),
        ([1.0], [NAN, NAN], "test"),
        ([NAN], [NAN], "training"),
    ],
    ids=["train-empty", "test-empty", "both-empty"],
)
def test_delete_refuses_to_empty_either_subset(train_x: list[float], test_x: list[float], emptied: str) -> None:
    """DELETE raises ``ValueError`` naming the subset it would empty, checking training first."""
    with pytest.raises(ValueError, match=f"{emptied} subset empty"):
        process_nan_values(DataFrame({"x": train_x}), DataFrame({"x": test_x}), ["x"], NanStrategy.DELETE)


@pytest.mark.xfail(strict=True, reason="fillna() puts the number among the text and turns the column into object")
def test_constant_on_text_column_is_rejected() -> None:
    """A numeric constant cannot fill a text column, so CONSTANT rejects it like MEAN and MEDIAN do."""
    train = DataFrame({"label": pd.array(["a", None, "c"], dtype="str")})
    test = DataFrame({"label": pd.array([None, "d"], dtype="str")})
    with pytest.raises(ValueError, match="not numeric"):
        process_nan_values(train, test, ["label"], NanStrategy.CONSTANT, -1.5)


# ----------------------------------------------------------------------------
# parse_constant(): text Python accepts but a user would not type as a number
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["1e309", "-1e309"])
def test_parse_constant_rejects_overflow(text: str) -> None:
    """A number too large for a float becomes infinity, so it is rejected as not finite."""
    with pytest.raises(ValueError, match="finite"):
        parse_constant(text)


@pytest.mark.xfail(strict=True, reason="float() accepts underscores between digits, so '1_000' is read as 1000")
def test_parse_constant_rejects_underscores() -> None:
    """Underscores are Python syntax, not a number format a user would type."""
    with pytest.raises(ValueError):
        parse_constant("1_000")
