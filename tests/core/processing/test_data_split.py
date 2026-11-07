"""Tests for splitting a dataset into training and test subsets."""

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
