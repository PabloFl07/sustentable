"""Tests for the dataset window.

The file dialog is not opened. ``show_dataset`` is the load the button runs after a path is chosen.
A cancelled dialog is not part of these tests.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pandas import NA, DataFrame, Series

from sustentable.data_import import CSVFileCheck, ExcelFileCheck, SQLiteFileCheck, get_reader
from sustentable.dataset_window import DATASET_EXTENSIONS, DatasetWindow, grid_records
from sustentable.exceptions import CorruptedFileError


def _window() -> DatasetWindow:
    """Return a window whose controls exist, without opening it."""
    window = DatasetWindow()
    window.build()
    return window


def test_dialog_extensions_follow_the_import_module():
    """The file dialog offers every extension the reader accepts, including those required by US-14."""
    assert DATASET_EXTENSIONS == (
        *CSVFileCheck.extensions,
        *ExcelFileCheck.extensions,
        *SQLiteFileCheck.extensions,
    )
    for extension in (".csv", ".xlsx", ".xls", ".sqlite", ".db"):
        assert extension in DATASET_EXTENSIONS


@pytest.mark.parametrize("fixture_name", ["example_csv", "example_xlsx", "example_sqlite"])
def test_show_dataset_displays_the_file(request, tmp_path, fixture_name):
    """A valid file shows its absolute path and every cell, including accents and ñ.

    Age stays a number, so the grid aligns that column to the right.
    """
    source = request.getfixturevalue(fixture_name)
    path = tmp_path / "carpeta con espacios" / f"cañón{source.suffix}"
    path.parent.mkdir()
    path.write_bytes(source.read_bytes())

    window = _window()
    window.show_dataset(path)

    assert window.dataset is not None
    assert window.dataset.equals(get_reader(path).read())
    assert window.path_text.text == str(path.resolve())
    assert "cañón" in window.path_text.text
    assert window.status_text.text == ""
    assert [(column.title, column.kind, column.align) for column in window.dataset_grid.columns] == [
        ("name", "text", "left"),
        ("age", "number", "right"),
    ]
    assert [row.cells for row in window.dataset_grid.rows] == [("José", 19), ("Begoña", 21)]


def test_grid_records_right_aligns_real_numbers():
    """Integers and real floats become number columns. Text, booleans, dates, and mixed values stay text."""
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

    columns, rows = grid_records(frame)
    _window().dataset_grid.set_data(columns=columns, rows=rows)

    assert [(column.title, column.kind, column.align) for column in columns] == [
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
    assert rows[0].cells[0] == "Ana"
    assert rows[1].cells[0] is None
    assert rows[0].cells[1:5] == (19, 1.5, None, "True")
    assert rows[1].cells[1] == 21
    assert rows[1].cells[2] is None
    assert rows[1].cells[3] is None
    assert isinstance(rows[0].cells[5], str) and "2020-01-02" in rows[0].cells[5]
    assert rows[1].cells[5] is None
    assert rows[0].cells[6] == "3"
    assert rows[1].cells[6] == "x"
    assert rows[0].cells[7] == 4 and type(rows[0].cells[7]) is int
    assert rows[1].cells[7] is None
    assert rows[0].cells[8] == 7 and type(rows[0].cells[8]) is int
    assert rows[1].cells[8] == float(np.float32(1.25)) and type(rows[1].cells[8]) is float


def test_grid_records_keeps_duplicate_headers_apart():
    """Two columns with the same header are still typed from their own values."""
    frame = DataFrame([["Ana", 19], ["Bo", 21]], columns=["value", "value"])

    columns, rows = grid_records(frame)

    assert [column.kind for column in columns] == ["text", "number"]
    assert [row.cells for row in rows] == [("Ana", 19), ("Bo", 21)]


def test_failed_load_keeps_the_previous_dataset(example_csv, tmp_path):
    """A damaged file shows the reader's message and leaves the path, the table, and ``dataset`` in place."""
    window = _window()
    window.show_dataset(example_csv)
    previous = window.dataset.copy()
    previous_path = window.path_text.text
    previous_rows = window.dataset_grid.rows

    # A different name from example_csv: both fixtures would otherwise share tmp_path / "example.csv".
    damaged = tmp_path / "damaged.csv"
    damaged.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe" * 20)
    with pytest.raises(CorruptedFileError) as caught:
        get_reader(damaged).read()
    window.show_dataset(damaged)

    assert window.status_text.text == str(caught.value)
    assert window.path_text.text == previous_path
    assert window.dataset.equals(previous)
    assert window.dataset_grid.rows == previous_rows


def test_show_dataset_accepts_a_relative_path(monkeypatch, tmp_path, example_csv):
    """A relative path is read from the working directory and shown as an absolute path."""
    monkeypatch.chdir(tmp_path)
    path = Path("ejemplo.csv")
    path.write_bytes(example_csv.read_bytes())

    window = _window()
    window.show_dataset(path)

    assert window.path_text.text == str(path.resolve())
    assert Path(window.path_text.text).is_absolute()
