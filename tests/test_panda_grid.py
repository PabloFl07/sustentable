"""Tests for converting dataframes into a reusable, atomic grid snapshot."""

import asyncio
from datetime import datetime
from threading import Event

import numpy as np
import pytest
from pandas import NA, DataFrame, NaT, Series
from pandas.testing import assert_frame_equal
from pysual import DataGrid, GridColumn, GridRow

from sustentable import panda_grid
from sustentable.panda_grid import PandaGrid


class _BrokenText:
    """Represent a value whose display conversion fails."""

    def __str__(self) -> str:
        raise ValueError("Cannot render this value.")


def test_constructor_loads_a_frame_and_forwards_control_properties() -> None:
    """The custom control is a DataGrid and accepts normal layout properties."""
    grid = PandaGrid(DataFrame({"age": [19, 21]}), flex=1, min_height=240)

    assert isinstance(grid, DataGrid)
    assert grid.flex == 1
    assert grid.min_height == 240
    assert [row.cells for row in grid.rows] == [(19,), (21,)]


def test_constructor_without_a_frame_accepts_native_grid_records() -> None:
    """Omitting a dataframe preserves the normal DataGrid constructor."""
    column = GridColumn("name", "Name")
    row = GridRow("person", ("Ana",))

    grid = PandaGrid(columns=(column,), rows=(row,))

    assert grid.columns == (column,)
    assert grid.rows == (row,)


@pytest.mark.parametrize("property_name", ["columns", "rows"])
def test_constructor_rejects_records_alongside_a_frame(property_name: str) -> None:
    """A supplied dataframe has one unambiguous source of columns and rows."""
    with pytest.raises(TypeError, match="either a dataframe"):
        PandaGrid(DataFrame({"age": [19]}), **{property_name: ()})


def test_real_numbers_use_numeric_columns_and_native_scalar_values() -> None:
    """Real numbers align right while booleans, dates, and mixed columns stay text."""
    frame = DataFrame(
        {
            "name": ["Ana", None],
            "age": [19, 21],
            "price": [1.5, float("nan")],
            "limit": [float("inf"), float("-inf")],
            "ok": [True, False],
            "when": [datetime(2020, 1, 2), None],
            "mixed": [3, "x"],
        }
    )
    frame["count"] = Series([4, NA], dtype="Int64")
    frame["raw"] = Series([np.int64(7), np.float32(1.25)], dtype=object)

    grid = PandaGrid(frame)

    assert [(column.title, column.kind, column.align) for column in grid.columns] == [
        ("name", "text", "left"),
        ("age", "number", "right"),
        ("price", "number", "right"),
        ("limit", "number", "right"),
        ("ok", "text", "left"),
        ("when", "text", "left"),
        ("mixed", "text", "left"),
        ("count", "number", "right"),
        ("raw", "number", "right"),
    ]
    first, second = grid.rows
    assert first.cells[:5] == ("Ana", 19, 1.5, None, "True")
    assert second.cells[:5] == (None, 21, None, None, "False")
    assert isinstance(first.cells[5], str) and "2020-01-02" in first.cells[5]
    assert second.cells[5] is None
    assert first.cells[6:] == ("3", 4, 7)
    assert second.cells[6:] == ("x", None, 1.25)
    assert type(first.cells[7]) is int
    assert type(first.cells[8]) is int
    assert type(second.cells[8]) is float


def test_missing_scalar_values_become_empty_cells() -> None:
    """Pandas and NumPy missing sentinels never become their textual spellings."""
    frame = DataFrame({"value": Series([None, NA, NaT, np.float32("nan"), np.datetime64("NaT", "ns")], dtype=object)})

    grid = PandaGrid(frame)

    assert [row.cells for row in grid.rows] == [(None,)] * len(frame)


def test_nonfinite_numpy_numbers_in_a_mixed_column_become_empty_cells() -> None:
    """Nonfinite numbers are missing even when text forces the column to be text."""
    frame = DataFrame({"value": Series([np.float32("inf"), "x", np.float64("-inf"), np.float32("nan")], dtype=object)})

    grid = PandaGrid(frame)

    assert grid.columns[0].kind == "text"
    assert [row.cells for row in grid.rows] == [(None,), ("x",), (None,), (None,)]


def test_duplicate_headers_and_indexes_keep_distinct_positions() -> None:
    """Duplicate dataframe labels do not merge columns or produce duplicate row keys."""
    frame = DataFrame([["Ana", 19], ["Bo", 21]], columns=["value", "value"], index=["person", "person"])

    grid = PandaGrid(frame)

    assert [(column.key, column.title, column.kind) for column in grid.columns] == [
        ("c0", "value", "text"),
        ("c1", "value", "number"),
    ]
    assert [(row.key, row.cells) for row in grid.rows] == [("0", ("Ana", 19)), ("1", ("Bo", 21))]


def test_headers_are_normalized_to_one_line() -> None:
    """Whitespace is collapsed and non-string column labels remain displayable."""
    grid = PandaGrid(DataFrame([[1, 2, 3]], columns=["  first\n\tlast  ", " \t\n", 42]))

    assert [column.title for column in grid.columns] == ["first last", "", "42"]


def test_loading_a_new_frame_replaces_all_previous_records() -> None:
    """Reloading uses the new schema and rows without retaining old cells."""
    grid = PandaGrid(DataFrame({"name": ["Ana", "Bo"], "age": [19, 21]}))

    grid.load_dataframe(DataFrame({"score": [1.5]}))

    assert [(column.key, column.title, column.kind) for column in grid.columns] == [("c0", "score", "number")]
    assert [(row.key, row.cells) for row in grid.rows] == [("0", (1.5,))]


@pytest.mark.parametrize(
    ("frame", "titles", "cells"),
    [
        (DataFrame(), [], []),
        (DataFrame({"age": Series([], dtype="Int64")}), ["age"], []),
        (DataFrame(index=["first", "second"]), [], [(), ()]),
    ],
)
def test_empty_frames_replace_previous_data(frame: DataFrame, titles: list[str], cells: list[tuple[object, ...]]) -> None:
    """Empty rows or columns are supported and clear the corresponding old data."""
    grid = PandaGrid(DataFrame({"old": [1]}))

    grid.load_dataframe(frame)

    assert [column.title for column in grid.columns] == titles
    assert [row.cells for row in grid.rows] == cells
    assert [row.key for row in grid.rows] == [str(index) for index in range(len(cells))]


@pytest.mark.parametrize("invalid_frame", [None, {"age": [19]}, [[19]], Series([19])])
def test_loading_invalid_input_keeps_previous_records(invalid_frame: object) -> None:
    """The public loader requires a dataframe without changing existing data on failure."""
    grid = PandaGrid(DataFrame({"age": [19]}))
    previous_columns, previous_rows = grid.columns, grid.rows

    with pytest.raises(TypeError, match="DataFrame"):
        grid.load_dataframe(invalid_frame)

    assert grid.columns == previous_columns
    assert grid.rows == previous_rows


def test_snapshot_does_not_mutate_or_retain_live_frame_values() -> None:
    """Loading leaves the frame intact and subsequent frame mutations cannot alter cells."""
    frame = DataFrame({"name": ["Ana"], "age": [19]})
    original = frame.copy(deep=True)

    grid = PandaGrid(frame)

    assert_frame_equal(frame, original)
    frame.iloc[0, 0] = "Bo"
    frame.iloc[0, 1] = 21
    frame.columns = ["changed", "other"]
    assert [column.title for column in grid.columns] == ["name", "age"]
    assert grid.rows[0].cells == ("Ana", 19)


def test_conversion_failure_keeps_records_selection_and_sorting() -> None:
    """A late cell conversion failure cannot publish a partially converted table."""
    grid = PandaGrid(DataFrame({"age": [19, 21]}))
    grid.update(selected_key="1", selected_column=0, sort_key="c0", sort_descending=True)
    previous_columns, previous_rows = grid.columns, grid.rows

    with pytest.raises(ValueError, match="Cannot render this value"):
        grid.load_dataframe(DataFrame({"different": ["valid", _BrokenText()]}))

    assert grid.columns == previous_columns
    assert grid.rows == previous_rows
    assert grid.selected_key == "1"
    assert grid.selected_column == 0
    assert grid.sort_key == "c0"
    assert grid.sort_descending is True


@pytest.mark.parametrize("invalid_frame", [DataFrame({"text": ["x" * 4097]}), DataFrame([range(641)])])
def test_native_grid_validation_failure_keeps_the_previous_table(invalid_frame: DataFrame) -> None:
    """Native limits on text length and column count fail atomically."""
    grid = PandaGrid(DataFrame({"age": [19, 21]}))
    grid.update(selected_key="1", selected_column=0)
    previous_columns, previous_rows = grid.columns, grid.rows

    with pytest.raises(ValueError):
        grid.load_dataframe(invalid_frame)

    assert grid.columns == previous_columns
    assert grid.rows == previous_rows
    assert grid.selected_key == "1"
    assert grid.selected_column == 0


def test_text_at_the_native_length_limit_is_preserved() -> None:
    """The adapter does not truncate valid text or reject the allowed boundary."""
    text = "ñ" * 4096

    grid = PandaGrid(DataFrame({"text": [text]}))

    assert grid.rows[0].cells == (text,)


def test_async_conversion_leaves_grid_controls_responsive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparation runs outside the UI owner, while the old snapshot remains usable until replacement."""
    grid = PandaGrid(DataFrame({"age": [19, 21]}))
    previous_columns, previous_rows = grid.columns, grid.rows
    replacement = DataFrame({"score": [1.5]})
    started, release, finished = Event(), Event(), Event()
    convert = panda_grid._grid_records

    def blocked_conversion(frame: DataFrame) -> tuple[tuple[GridColumn, ...], tuple[GridRow, ...]]:
        started.set()
        try:
            assert release.wait(5), "The test did not release grid conversion."
            return convert(frame)
        finally:
            finished.set()

    monkeypatch.setattr(panda_grid, "_grid_records", blocked_conversion)

    async def exercise() -> None:
        pending = asyncio.create_task(grid.load_dataframe_async(replacement))
        try:
            assert await asyncio.to_thread(started.wait, 5), "Grid conversion did not start."
            # Changing selection must reach Pysual's UI owner before conversion can finish.
            grid.selected_key = "1"
            assert grid.selected_key == "1"
            assert not finished.is_set()
            assert grid.columns == previous_columns
            assert grid.rows == previous_rows
            release.set()
            await asyncio.wait_for(pending, 5)
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            if started.is_set():
                assert await asyncio.to_thread(finished.wait, 5)

    asyncio.run(exercise())

    assert [column.title for column in grid.columns] == ["score"]
    assert [row.cells for row in grid.rows] == [(1.5,)]


@pytest.mark.parametrize("invalid_frame", [DataFrame({"text": ["x" * 4097]}), DataFrame({"value": ["valid", _BrokenText()]})])
def test_async_conversion_or_validation_failure_keeps_previous_data(invalid_frame: DataFrame) -> None:
    """Failures from both worker conversion and UI validation propagate without replacing the table."""
    grid = PandaGrid(DataFrame({"age": [19, 21]}))
    grid.update(selected_key="1", selected_column=0, sort_key="c0", sort_descending=True)
    previous_columns, previous_rows = grid.columns, grid.rows

    with pytest.raises(ValueError):
        asyncio.run(grid.load_dataframe_async(invalid_frame))

    assert grid.columns == previous_columns
    assert grid.rows == previous_rows
    assert grid.selected_key == "1"
    assert grid.sort_key == "c0"
    assert grid.sort_descending


def test_cancelled_async_conversion_cannot_overwrite_a_newer_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled conversion may finish in its worker, but its result is never committed."""
    cancelled_frame = DataFrame({"stale": [99]})
    replacement_frame = DataFrame({"current": [42]})
    cancelled_complete = Event()

    class _ObservedGrid(PandaGrid):
        """Signal when cancellation has actually finished on Pysual's UI owner."""

        async def load_dataframe_async(self, frame: DataFrame) -> None:
            """Observe completion after the real async grid loader has returned."""
            try:
                await super().load_dataframe_async(frame)
            finally:
                if frame is cancelled_frame:
                    cancelled_complete.set()

    grid = _ObservedGrid(DataFrame({"old": [1]}))
    started, release, finished = Event(), Event(), Event()
    convert = panda_grid._grid_records

    def controlled_conversion(frame: DataFrame) -> tuple[tuple[GridColumn, ...], tuple[GridRow, ...]]:
        if frame is not cancelled_frame:
            return convert(frame)
        started.set()
        try:
            assert release.wait(5), "The test did not release the cancelled conversion."
            return convert(frame)
        finally:
            finished.set()

    monkeypatch.setattr(panda_grid, "_grid_records", controlled_conversion)

    async def exercise() -> None:
        pending = asyncio.create_task(grid.load_dataframe_async(cancelled_frame))
        try:
            assert await asyncio.to_thread(started.wait, 5), "Grid conversion did not start."
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert await asyncio.to_thread(cancelled_complete.wait, 5), "The cancelled UI conversion did not finish."
            assert [row.cells for row in grid.rows] == [(1,)]
            grid.load_dataframe(replacement_frame)
            release.set()
            assert await asyncio.to_thread(finished.wait, 5), "The cancelled conversion did not finish."
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            if started.is_set():
                assert await asyncio.to_thread(finished.wait, 5)

    asyncio.run(exercise())

    assert [column.title for column in grid.columns] == ["current"]
    assert [row.cells for row in grid.rows] == [(42,)]
