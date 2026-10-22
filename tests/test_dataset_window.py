"""Tests for the dataset window.

The native file dialog is mocked; file reads and PandaGrid conversion use their real implementations.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pysual import CapabilityError, ClickEvent

from sustentable.data_import import CSVFileCheck, ExcelFileCheck, SQLiteFileCheck, get_reader
from sustentable.dataset_window import DATASET_EXTENSIONS, DatasetWindow
from sustentable.exceptions import CorruptedFileError
from sustentable.panda_grid import PandaGrid


def _window() -> DatasetWindow:
    """Return a window whose controls exist, without opening it."""
    window = DatasetWindow()
    window.build()
    return window


def test_dialog_extensions_follow_the_import_module() -> None:
    """The file dialog offers every extension the reader accepts, including those required by US-14."""
    assert DATASET_EXTENSIONS == (
        *CSVFileCheck.extensions,
        *ExcelFileCheck.extensions,
        *SQLiteFileCheck.extensions,
    )
    for extension in (".csv", ".xlsx", ".xls", ".sqlite", ".db"):
        assert extension in DATASET_EXTENSIONS


@pytest.mark.parametrize("fixture_name", ["example_csv", "example_xlsx", "example_sqlite"])
def test_show_dataset_displays_the_file(request: pytest.FixtureRequest, tmp_path: Path, fixture_name: str) -> None:
    """A valid file shows its absolute path and every cell, including accents and ñ.

    Age stays a number, so the grid aligns that column to the right.
    """
    source = request.getfixturevalue(fixture_name)
    path = tmp_path / "carpeta con espacios" / f"cañón{source.suffix}"
    path.parent.mkdir()
    path.write_bytes(source.read_bytes())

    window = _window()
    window.show_dataset(path)

    assert isinstance(window.dataset_grid, PandaGrid)
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


def test_failed_load_keeps_the_previous_dataset(example_csv: Path, tmp_path: Path) -> None:
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


def test_show_dataset_accepts_a_relative_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, example_csv: Path) -> None:
    """A relative path is read from the working directory and shown as an absolute path."""
    monkeypatch.chdir(tmp_path)
    path = Path("ejemplo.csv")
    path.write_bytes(example_csv.read_bytes())

    window = _window()
    window.show_dataset(path)

    assert window.path_text.text == str(path.resolve())
    assert Path(window.path_text.text).is_absolute()


def test_new_dataset_replaces_the_table_and_clears_an_error(example_csv: Path, tmp_path: Path) -> None:
    """A successful reload updates the schema, path, and raw frame after a failed load."""
    window = _window()
    window.show_dataset(example_csv)
    window.show_dataset(tmp_path / "missing.csv")
    assert window.status_text.text
    replacement = tmp_path / "replacement.csv"
    replacement.write_text("score\n1.5\n", encoding="utf-8")

    window.show_dataset(replacement)

    assert window.dataset.equals(get_reader(replacement).read())
    assert window.path_text.text == str(replacement.resolve())
    assert window.status_text.text == ""
    assert [column.title for column in window.dataset_grid.columns] == ["score"]
    assert [row.cells for row in window.dataset_grid.rows] == [(1.5,)]


def test_grid_validation_error_preserves_the_previous_dataset(example_csv: Path, tmp_path: Path) -> None:
    """A readable file whose cell exceeds the grid limit leaves all previously loaded data intact."""
    window = _window()
    window.show_dataset(example_csv)
    previous = window.dataset
    previous_path = window.path_text.text
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    oversized = tmp_path / "oversized.csv"
    oversized.write_text("text\n" + "x" * 4097 + "\n", encoding="utf-8")

    window.show_dataset(oversized)

    assert "4096" in window.status_text.text
    assert window.dataset is previous
    assert window.path_text.text == previous_path
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows


def test_filesystem_error_is_reported_without_losing_data(example_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An I/O failure during a read is shown without discarding the previous successful load."""
    window = _window()
    window.show_dataset(example_csv)
    previous = window.dataset
    previous_rows = window.dataset_grid.rows
    reader = Mock()
    reader.read.side_effect = OSError("The file became unavailable.")
    monkeypatch.setattr("sustentable.dataset_window.get_reader", Mock(return_value=reader))

    window.show_dataset(example_csv)

    assert window.status_text.text == "The file became unavailable."
    assert window.dataset is previous
    assert window.path_text.text == str(example_csv.resolve())
    assert window.dataset_grid.rows == previous_rows


def test_cancelled_dialog_preserves_the_entire_state(example_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling does not read a file, clear an error, or change the displayed data."""
    window = _window()
    window.show_dataset(example_csv)
    window.status_text.text = "Previous error"
    previous = window.dataset
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    dialog = AsyncMock(return_value=None)
    reader = Mock()
    monkeypatch.setattr(DatasetWindow, "choose_path", dialog)
    monkeypatch.setattr("sustentable.dataset_window.get_reader", reader)

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    dialog.assert_awaited_once_with(title="Open dataset", extensions=DATASET_EXTENSIONS)
    reader.assert_not_called()
    assert window.dataset is previous
    assert window.path_text.text == str(example_csv.resolve())
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows
    assert window.status_text.text == "Previous error"


def test_dialog_selection_loads_the_chosen_file(example_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The button handler loads the selected path using the real importer and PandaGrid."""
    window = _window()
    monkeypatch.setattr(DatasetWindow, "choose_path", AsyncMock(return_value=example_csv))

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    assert window.dataset.equals(get_reader(example_csv).read())
    assert window.path_text.text == str(example_csv.resolve())
    assert [row.cells for row in window.dataset_grid.rows] == [("José", 19), ("Begoña", 21)]


def test_dialog_error_is_reported(example_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A host without a usable file dialog reports the error and keeps its existing data."""
    window = _window()
    window.show_dataset(example_csv)
    previous = window.dataset
    monkeypatch.setattr(DatasetWindow, "choose_path", AsyncMock(side_effect=CapabilityError("File dialog unavailable.")))

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    assert window.dataset is previous
    assert window.path_text.text == str(example_csv.resolve())
    assert window.status_text.text == "File dialog unavailable."
