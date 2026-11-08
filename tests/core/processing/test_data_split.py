"""Tests for splitting a dataset into training and test subsets."""

import math
import random

import numpy as np
import pandas as pd
import pytest

from sustentable.core.processing.data_split import count_test_rows, split_dataset


@pytest.fixture
def frame() -> pd.DataFrame:
    """Ten rows with a unique id to track where each row ends up."""
    return pd.DataFrame({"id": range(10), "value": [x * 10 for x in range(10)]})


def test_returns_train_before_test(frame: pd.DataFrame) -> None:
    """The first subset is the larger training one."""
    train, test = split_dataset(frame, 0.2, seed=42)

    assert (len(train), len(test)) == (8, 2)


@pytest.mark.parametrize(("fraction", "expected_test"), [(0.2, 2), (0.25, 3), (0.04, 1), (0.5, 5)])
def test_sizes_match_count_test_rows(frame: pd.DataFrame, fraction: float, expected_test: int) -> None:
    """The preview count and the real split agree, including the upward rounding."""
    train, test = split_dataset(frame, fraction, seed=42)

    assert count_test_rows(len(frame), fraction) == len(test) == expected_test
    assert len(train) == len(frame) - expected_test


def test_keeps_every_row_once_and_all_columns(frame: pd.DataFrame) -> None:
    """No row is lost or duplicated, and both subsets keep the original columns."""
    train, test = split_dataset(frame, 0.3, seed=42)

    assert sorted([*train.id, *test.id]) == list(range(10))
    assert list(train.columns) == list(test.columns) == list(frame.columns)


def test_rows_are_shuffled(frame: pd.DataFrame) -> None:
    """The training subset is not simply the first rows of the dataset."""
    train, _ = split_dataset(frame, 0.2, seed=42)

    assert list(train.id) != list(range(8))


def test_same_seed_gives_same_split_and_different_seed_changes_it() -> None:
    """The seed makes the split reproducible."""
    frame = pd.DataFrame({"id": range(100)})

    first_train, first_test = split_dataset(frame, 0.5, seed=1)
    again_train, again_test = split_dataset(frame, 0.5, seed=1)
    other_train, _ = split_dataset(frame, 0.5, seed=2)

    pd.testing.assert_frame_equal(first_train, again_train)
    pd.testing.assert_frame_equal(first_test, again_test)
    assert not first_train.equals(other_train)


def test_indices_are_reset_and_input_is_untouched(frame: pd.DataFrame) -> None:
    """Each subset is indexed from zero and changing it does not change the input."""
    original = frame.copy(deep=True)

    train, test = split_dataset(frame, 0.3, seed=42)
    train.loc[0, "value"] = -999

    assert list(train.index) == list(range(len(train)))
    assert list(test.index) == list(range(len(test)))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_fraction_outside_the_open_interval_is_rejected(frame: pd.DataFrame, fraction: float) -> None:
    """Both subsets must be able to receive rows."""
    with pytest.raises(ValueError):
        split_dataset(frame, fraction, seed=42)


def test_split_that_would_leave_training_empty_is_rejected() -> None:
    """A single row cannot be split into two non-empty subsets."""
    with pytest.raises(ValueError):
        split_dataset(pd.DataFrame({"id": [1]}), 0.5, seed=42)


# ----------------------------------------------------------------------------
# Reproducibility: only the seed argument matters
# ----------------------------------------------------------------------------


def test_split_ignores_global_random_state(frame: pd.DataFrame) -> None:
    """Seeding or advancing the global generators of ``random`` and NumPy does not change the split."""
    random.seed(0)
    np.random.seed(0)
    first_train, first_test = split_dataset(frame, 0.3, seed=42)

    random.seed(123)
    np.random.seed(123)
    np.random.rand(50)
    again_train, again_test = split_dataset(frame, 0.3, seed=42)

    pd.testing.assert_frame_equal(first_train, again_train)
    pd.testing.assert_frame_equal(first_test, again_test)


def test_split_does_not_change_global_random_state(frame: pd.DataFrame) -> None:
    """Splitting leaves the global generators as they were, so it cannot affect other random code."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    split_dataset(frame, 0.3, seed=42)

    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    assert np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


# ----------------------------------------------------------------------------
# Minimum sizes and limits: both subsets always get at least one row
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "fraction", "expected_train", "expected_test"),
    [
        (2, 0.5, 1, 1),  # The smallest dataset that can be split.
        (2, 0.01, 1, 1),  # A tiny fraction still gives the test subset one row.
        (10, 0.01, 9, 1),
        (10, 0.9, 1, 9),  # The largest fraction that leaves one training row.
        (10, 0.99, None, None),  # Rounding up would leave training empty.
        (0, 0.5, None, None),  # An empty dataset cannot be split.
    ],
    ids=["2-rows-half", "2-rows-tiny", "10-rows-tiny", "10-rows-90", "10-rows-99", "empty"],
)
def test_split_limits(rows: int, fraction: float, expected_train: int | None, expected_test: int | None) -> None:
    """At the limits each subset keeps at least one row, or the split is rejected with ``ValueError``."""
    frame = pd.DataFrame({"id": range(rows)})
    if expected_test is None:
        with pytest.raises(ValueError):
            split_dataset(frame, fraction, seed=42)
        return

    train, test = split_dataset(frame, fraction, seed=42)
    assert (len(train), len(test)) == (expected_train, expected_test)


def test_valid_splits_never_leave_a_subset_empty() -> None:
    """For every dataset of 2 to 30 rows and every whole percentage, an accepted split fills both subsets."""
    for rows in range(2, 31):
        frame = pd.DataFrame({"id": range(rows)})
        for percent in range(1, 100):
            fraction = percent / 100
            if count_test_rows(rows, fraction) >= rows:
                continue
            train, test = split_dataset(frame, fraction, seed=42)
            assert len(train) >= 1
            assert len(test) >= 1
            assert len(train) + len(test) == rows


# ----------------------------------------------------------------------------
# Rounding: the test subset rounds up, never to the nearest
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "fraction", "expected_test"),
    [(10, 0.21, 3), (10, 0.29, 3), (3, 0.34, 2), (3, 0.33, 1), (7, 0.5, 4)],
)
def test_test_subset_rounds_up(rows: int, fraction: float, expected_test: int) -> None:
    """A fractional number of test rows is always rounded up, even when it is just above a whole number."""
    _, test = split_dataset(pd.DataFrame({"id": range(rows)}), fraction, seed=42)

    assert count_test_rows(rows, fraction) == len(test) == expected_test


@pytest.mark.xfail(
    strict=True,
    reason="percent / 100 * rows has floating point error (0.07 * 100 = 7.000000000000001), so ceil() adds an extra row",
)
@pytest.mark.parametrize(("rows", "percent"), [(100, 7), (50, 14), (25, 28)])
def test_exact_percentage_is_not_rounded_up(rows: int, percent: int) -> None:
    """When the percentage gives a whole number of rows, as the interface passes it (``percent / 100``), no row is added."""
    _, test = split_dataset(pd.DataFrame({"id": range(rows)}), percent / 100, seed=42)

    expected = math.ceil(percent * rows / 100)
    assert count_test_rows(rows, percent / 100) == len(test) == expected
