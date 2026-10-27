"""Tests for the dataset window.

File dialogs and error presentation are mocked; file reads and PandaGrid conversion use their real implementations.
"""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pandas import DataFrame
from pysual import CapabilityError, ClickEvent, GridColumn, GridRow
from pysual.errors import LifecycleError

from sustentable.data.exceptions import CorruptedFileError, MissingFileError, UnsupportedExtensionError
from sustentable.data.readers import CSVFileCheck, ExcelFileCheck, SQLiteFileCheck, get_reader
from sustentable.gui.controls import panda_grid
from sustentable.gui.controls.loading_spinner import LoadingSpinner
from sustentable.gui.controls.panda_grid import PandaGrid
from sustentable.gui.dataset_window import DATASET_EXTENSIONS, DatasetWindow
from sustentable.gui.error_dialog import ErrorDialog


@dataclass
class _SpinnerActivity:
    """Observe entry and completion while leaving the real animation unchanged."""

    started: Event = field(default_factory=Event)
    completed: Event = field(default_factory=Event)
    runs: int = 0


@pytest.fixture
def spinner_activity(monkeypatch: pytest.MonkeyPatch) -> _SpinnerActivity:
    """Observe the UI-owned animation without replacing its timing or phase updates."""
    activity = _SpinnerActivity()

    class _ObservedSpinner(LoadingSpinner):
        """Report lifecycle events around the actual spinner coroutine."""

        async def animate(self) -> None:
            """Delegate animation and report when its coroutine finishes on the UI owner."""
            activity.runs += 1
            activity.completed.clear()
            activity.started.set()
            try:
                await super().animate()
            finally:
                activity.completed.set()

    monkeypatch.setattr("sustentable.gui.dataset_window.LoadingSpinner", _ObservedSpinner)
    return activity


async def _wait_for_spinner_advance(spinner: LoadingSpinner) -> None:
    """Wait for an observable phase change with a bounded condition check."""
    initial_phase = spinner.phase
    async with asyncio.timeout(5):
        while spinner.phase == initial_phase:
            await asyncio.sleep(0.01)


def _window() -> DatasetWindow:
    """Return a window whose controls exist, without opening it."""
    window = DatasetWindow()
    window.build()
    return window


@pytest.fixture
def show_error_dialog() -> Iterator[Mock]:
    """Capture the window's requests to its reusable error control."""
    with patch.object(ErrorDialog, "show_error", autospec=True) as show:
        yield show


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
def test_show_dataset_displays_the_file(request: pytest.FixtureRequest, tmp_path: Path, fixture_name: str, show_error_dialog: Mock) -> None:
    """A valid file shows its absolute path and every cell, including accents and ñ.

    Age stays a number, so the grid aligns that column to the right.
    """
    source = request.getfixturevalue(fixture_name)
    path = tmp_path / "carpeta con espacios" / f"cañón{source.suffix}"
    path.parent.mkdir()
    path.write_bytes(source.read_bytes())

    window = _window()
    asyncio.run(window.show_dataset(path))

    assert isinstance(window.dataset_grid, PandaGrid)
    assert window.dataset is not None
    assert window.dataset.equals(get_reader(path).read())
    assert window.path_text.text == str(path.resolve())
    assert "cañón" in window.path_text.text
    show_error_dialog.assert_not_called()
    assert [(column.title, column.kind, column.align) for column in window.dataset_grid.columns] == [
        ("name", "text", "left"),
        ("age", "number", "right"),
    ]
    assert [row.cells for row in window.dataset_grid.rows] == [("José", 19), ("Begoña", 21)]


def test_failed_load_keeps_the_previous_dataset(example_csv: Path, tmp_path: Path, show_error_dialog: Mock) -> None:
    """A damaged file shows the reader's message and leaves the path, the table, and ``dataset`` in place."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_path = window.path_text.text
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows

    # A different name from example_csv: both fixtures would otherwise share tmp_path / "example.csv".
    damaged = tmp_path / "damaged.csv"
    damaged.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe" * 20)
    with pytest.raises(CorruptedFileError) as caught:
        get_reader(damaged).read()
    asyncio.run(window.show_dataset(damaged))

    show_error_dialog.assert_called_once_with(window.error_dialog, str(caught.value))
    assert window.path_text.text == previous_path
    assert window.dataset is previous
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows


def test_show_dataset_accepts_a_relative_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, example_csv: Path, show_error_dialog: Mock
) -> None:
    """A relative path is read from the working directory and shown as an absolute path."""
    monkeypatch.chdir(tmp_path)
    path = Path("ejemplo.csv")
    path.write_bytes(example_csv.read_bytes())

    window = _window()
    asyncio.run(window.show_dataset(path))

    assert window.path_text.text == str(path.resolve())
    assert Path(window.path_text.text).is_absolute()
    show_error_dialog.assert_not_called()


def test_new_dataset_replaces_the_table_after_an_error(example_csv: Path, tmp_path: Path, show_error_dialog: Mock) -> None:
    """A successful reload updates the schema, path, and raw frame after a failed load."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    missing = tmp_path / "missing.csv"
    with pytest.raises(MissingFileError) as caught:
        get_reader(missing).read()
    asyncio.run(window.show_dataset(missing))
    show_error_dialog.assert_called_once_with(window.error_dialog, str(caught.value))
    show_error_dialog.reset_mock()
    replacement = tmp_path / "replacement.csv"
    replacement.write_text("score\n1.5\n", encoding="utf-8")

    asyncio.run(window.show_dataset(replacement))

    assert window.dataset.equals(get_reader(replacement).read())
    assert window.path_text.text == str(replacement.resolve())
    show_error_dialog.assert_not_called()
    assert [column.title for column in window.dataset_grid.columns] == ["score"]
    assert [row.cells for row in window.dataset_grid.rows] == [(1.5,)]


def test_grid_validation_error_preserves_the_previous_dataset(example_csv: Path, tmp_path: Path, show_error_dialog: Mock) -> None:
    """A readable file whose cell exceeds the grid limit leaves all previously loaded data intact."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_path = window.path_text.text
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    oversized = tmp_path / "oversized.csv"
    oversized.write_text("text\n" + "x" * 4097 + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        PandaGrid(get_reader(oversized).read())

    asyncio.run(window.show_dataset(oversized))

    show_error_dialog.assert_called_once_with(window.error_dialog, str(caught.value))
    assert window.dataset is previous
    assert window.path_text.text == previous_path
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows


def test_filesystem_error_is_reported_without_losing_data(example_csv: Path, monkeypatch: pytest.MonkeyPatch, show_error_dialog: Mock) -> None:
    """An I/O failure during a read is shown without discarding the previous successful load."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    reader = Mock()
    reader.read.side_effect = OSError("The file became unavailable.")
    monkeypatch.setattr("sustentable.gui.dataset_window.get_reader", Mock(return_value=reader))

    asyncio.run(window.show_dataset(example_csv))

    show_error_dialog.assert_called_once_with(window.error_dialog, "The file became unavailable.")
    assert window.dataset is previous
    assert window.path_text.text == str(example_csv.resolve())
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows


def test_cancelled_dialog_preserves_the_entire_state(example_csv: Path, monkeypatch: pytest.MonkeyPatch, show_error_dialog: Mock) -> None:
    """Cancelling does not read a file, present an error, or change the displayed data."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    dialog = AsyncMock(return_value=None)
    reader = Mock()
    monkeypatch.setattr(DatasetWindow, "choose_path", dialog)
    monkeypatch.setattr("sustentable.gui.dataset_window.get_reader", reader)

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    dialog.assert_awaited_once_with(title="Open dataset", extensions=DATASET_EXTENSIONS)
    reader.assert_not_called()
    assert window.dataset is previous
    assert window.path_text.text == str(example_csv.resolve())
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows
    show_error_dialog.assert_not_called()


def test_dialog_selection_loads_the_chosen_file(example_csv: Path, monkeypatch: pytest.MonkeyPatch, show_error_dialog: Mock) -> None:
    """The button handler loads the selected path using the real importer and PandaGrid."""
    window = _window()
    monkeypatch.setattr(DatasetWindow, "choose_path", AsyncMock(return_value=example_csv))

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    assert window.dataset.equals(get_reader(example_csv).read())
    assert window.path_text.text == str(example_csv.resolve())
    assert [row.cells for row in window.dataset_grid.rows] == [("José", 19), ("Begoña", 21)]
    show_error_dialog.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        CapabilityError("File dialog unavailable."),
        OSError("The file chooser could not access the selected folder."),
        ValueError("Choose a file ending in " + ", ".join(DATASET_EXTENSIONS)),
    ],
)
def test_dialog_error_preserves_data_and_allows_another_browse(
    example_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, show_error_dialog: Mock, error: Exception
) -> None:
    """A rejected selection reports its error without reading, then a later valid browse succeeds."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    replacement = tmp_path / "replacement.csv"
    replacement.write_text("score\n1.5\n", encoding="utf-8")
    dialog = AsyncMock(side_effect=[error, replacement])
    reader = Mock(wraps=get_reader)
    monkeypatch.setattr(DatasetWindow, "choose_path", dialog)
    monkeypatch.setattr("sustentable.gui.dataset_window.get_reader", reader)

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    dialog.assert_awaited_once_with(title="Open dataset", extensions=DATASET_EXTENSIONS)
    reader.assert_not_called()
    assert window.dataset is previous
    assert window.path_text.text == str(example_csv.resolve())
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows
    show_error_dialog.assert_called_once_with(window.error_dialog, str(error))
    show_error_dialog.reset_mock()

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    assert dialog.await_count == 2
    reader.assert_called_once_with(replacement)
    show_error_dialog.assert_not_called()
    assert window.dataset.equals(get_reader(replacement).read())
    assert window.path_text.text == str(replacement.resolve())
    assert [column.title for column in window.dataset_grid.columns] == ["score"]
    assert [row.cells for row in window.dataset_grid.rows] == [(1.5,)]


def test_unsupported_path_returned_by_dialog_is_reported(
    example_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, show_error_dialog: Mock
) -> None:
    """An unsupported Markdown path returned by the chooser is still rejected by the importer."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    unsupported = tmp_path / "notes.md"
    unsupported.write_text("# Not a dataset\n", encoding="utf-8")
    with pytest.raises(UnsupportedExtensionError) as caught:
        get_reader(unsupported).read()
    monkeypatch.setattr(DatasetWindow, "choose_path", AsyncMock(return_value=unsupported))

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    show_error_dialog.assert_called_once_with(window.error_dialog, str(caught.value))
    assert window.dataset is previous
    assert window.path_text.text == str(example_csv.resolve())
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows


def test_closed_window_during_dialog_preserves_data_without_an_error(
    example_csv: Path, monkeypatch: pytest.MonkeyPatch, show_error_dialog: Mock
) -> None:
    """A chooser interrupted by window teardown neither reads a file nor presents another dialog."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    reader = Mock()
    monkeypatch.setattr(DatasetWindow, "choose_path", AsyncMock(side_effect=LifecycleError("The owner was closed.")))
    monkeypatch.setattr("sustentable.gui.dataset_window.get_reader", reader)

    asyncio.run(window.open_dataset_on_click(ClickEvent(source=window.open_dataset)))

    reader.assert_not_called()
    show_error_dialog.assert_not_called()
    assert window.dataset is previous
    assert window.path_text.text == str(example_csv.resolve())
    assert window.dataset_grid.columns == previous_columns
    assert window.dataset_grid.rows == previous_rows


@pytest.mark.parametrize("read_error", [None, OSError("The dataset became unavailable.")])
def test_pending_load_keeps_controls_responsive_and_ignores_overlaps(
    example_csv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    show_error_dialog: Mock,
    read_error: OSError | None,
    spinner_activity: _SpinnerActivity,
) -> None:
    """A blocked reader leaves the UI usable, advertises loading, and restores controls on success or failure."""
    window = _window()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    replacement = tmp_path / "replacement.csv"
    frame = DataFrame({"score": [1.5]})
    started, release, finished = Event(), Event(), Event()
    requested_paths: list[str | Path] = []

    def blocked_read(path: str | Path) -> tuple[str, DataFrame]:
        requested_paths.append(path)
        started.set()
        try:
            assert release.wait(5), "The test did not release the dataset reader."
            if read_error is not None:
                raise read_error
            return str(Path(path).resolve()), frame
        finally:
            finished.set()

    monkeypatch.setattr("sustentable.gui.dataset_window._read_dataset", blocked_read)

    async def exercise() -> None:
        pending = asyncio.create_task(window.show_dataset(replacement))
        try:
            assert await asyncio.to_thread(started.wait, 5), "The dataset reader did not start."
            assert window.loading_row.visible
            assert not window.open_dataset.enabled
            assert window.dataset is previous
            assert window.path_text.text == str(example_csv.resolve())
            assert window.dataset_grid.rows == previous_rows
            # This public mutation must reach Pysual's UI owner while the worker is still blocked.
            window.path_label.text = "The UI remains responsive"
            assert window.path_label.text == "The UI remains responsive"
            assert not finished.is_set()
            await _wait_for_spinner_advance(window.loading_spinner)

            await asyncio.wait_for(window.show_dataset(tmp_path / "ignored.csv"), 5)
            assert requested_paths == [replacement]
            assert window.loading_row.visible
            assert not window.open_dataset.enabled
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

    assert not window.loading_row.visible
    assert window.open_dataset.enabled
    assert spinner_activity.completed.is_set()
    if read_error is None:
        show_error_dialog.assert_not_called()
        assert window.dataset is frame
        assert window.path_text.text == str(replacement.resolve())
        assert [row.cells for row in window.dataset_grid.rows] == [(1.5,)]
    else:
        show_error_dialog.assert_called_once_with(window.error_dialog, str(read_error))
        assert window.dataset is previous
        assert window.path_text.text == str(example_csv.resolve())
        assert window.dataset_grid.columns == previous_columns
        assert window.dataset_grid.rows == previous_rows


def test_cancelled_load_restores_controls_and_cannot_commit_late_results(
    example_csv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    show_error_dialog: Mock,
    spinner_activity: _SpinnerActivity,
) -> None:
    """Cancelling the await discards the worker's eventual result and permits a subsequent load."""
    cancelled_path = tmp_path / "cancelled.csv"
    replacement_path = tmp_path / "replacement.csv"
    cancelled_complete = Event()

    class _ObservedWindow(DatasetWindow):
        """Signal completion on the UI owner, after the real loader has cleaned up."""

        async def show_dataset(self, path: str | Path) -> None:
            """Delegate loading and expose when the cancelled UI invocation actually finishes."""
            try:
                await super().show_dataset(path)
            finally:
                if path == cancelled_path:
                    cancelled_complete.set()

    window = _ObservedWindow()
    window.build()
    asyncio.run(window.show_dataset(example_csv))
    previous = window.dataset
    previous_columns, previous_rows = window.dataset_grid.columns, window.dataset_grid.rows
    cancelled_frame = DataFrame({"stale": [99]})
    replacement_frame = DataFrame({"current": [42]})
    started, release, finished = Event(), Event(), Event()

    def controlled_read(path: str | Path) -> tuple[str, DataFrame]:
        if path != cancelled_path:
            return str(Path(path).resolve()), replacement_frame
        started.set()
        try:
            assert release.wait(5), "The test did not release the cancelled reader."
            return str(cancelled_path.resolve()), cancelled_frame
        finally:
            finished.set()

    monkeypatch.setattr("sustentable.gui.dataset_window._read_dataset", controlled_read)

    async def exercise() -> None:
        pending = asyncio.create_task(window.show_dataset(cancelled_path))
        try:
            assert await asyncio.to_thread(started.wait, 5), "The dataset reader did not start."
            await _wait_for_spinner_advance(window.loading_spinner)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            # Pysual forwards caller cancellation to its UI loop asynchronously.
            assert await asyncio.to_thread(cancelled_complete.wait, 5), "The cancelled UI load did not finish."
            assert not window.loading_row.visible
            assert window.open_dataset.enabled
            assert spinner_activity.completed.is_set()
            assert window.dataset is previous
            assert window.path_text.text == str(example_csv.resolve())
            assert window.dataset_grid.columns == previous_columns
            assert window.dataset_grid.rows == previous_rows

            await asyncio.wait_for(window.show_dataset(replacement_path), 5)
            assert window.dataset is replacement_frame
            release.set()
            assert await asyncio.to_thread(finished.wait, 5), "The cancelled worker did not finish."
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            if started.is_set():
                assert await asyncio.to_thread(finished.wait, 5)

    asyncio.run(exercise())

    show_error_dialog.assert_not_called()
    assert window.dataset is replacement_frame
    assert window.path_text.text == str(replacement_path.resolve())
    assert [row.cells for row in window.dataset_grid.rows] == [(42,)]
    assert not window.loading_row.visible
    assert window.open_dataset.enabled
    assert spinner_activity.completed.is_set()


def test_spinner_advances_during_reading_and_conversion_and_restarts_for_next_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, show_error_dialog: Mock, spinner_activity: _SpinnerActivity
) -> None:
    """The real ring advances during both worker stages and each completed load stops its animation."""
    window = _window()
    frame = DataFrame({"score": [1.5]})
    read_started, release_read, read_finished = Event(), Event(), Event()
    conversion_started, release_conversion, conversion_finished = Event(), Event(), Event()
    convert = panda_grid._grid_records

    def blocked_read(path: str | Path) -> tuple[str, DataFrame]:
        read_started.set()
        try:
            assert release_read.wait(5), "The test did not release the dataset reader."
            return str(Path(path).resolve()), frame
        finally:
            read_finished.set()

    def blocked_conversion(value: DataFrame) -> tuple[tuple[GridColumn, ...], tuple[GridRow, ...]]:
        conversion_started.set()
        try:
            assert release_conversion.wait(5), "The test did not release grid conversion."
            return convert(value)
        finally:
            conversion_finished.set()

    monkeypatch.setattr("sustentable.gui.dataset_window._read_dataset", blocked_read)
    monkeypatch.setattr(panda_grid, "_grid_records", blocked_conversion)

    async def exercise() -> None:
        for run in (1, 2):
            for event in (read_started, release_read, read_finished, conversion_started, release_conversion, conversion_finished):
                event.clear()
            spinner_activity.started.clear()
            pending = asyncio.create_task(window.show_dataset(tmp_path / f"dataset-{run}.csv"))
            try:
                assert await asyncio.to_thread(read_started.wait, 5), "The dataset reader did not start."
                assert await asyncio.to_thread(spinner_activity.started.wait, 5), "The spinner did not start."
                assert spinner_activity.runs == run
                assert window.loading_row.visible
                assert not window.open_dataset.enabled
                assert not spinner_activity.completed.is_set()
                await _wait_for_spinner_advance(window.loading_spinner)

                release_read.set()
                assert await asyncio.to_thread(conversion_started.wait, 5), "Grid conversion did not start."
                assert window.loading_row.visible
                await _wait_for_spinner_advance(window.loading_spinner)
                assert not conversion_finished.is_set()
                assert not spinner_activity.completed.is_set()
                release_conversion.set()
                await asyncio.wait_for(pending, 5)

                assert spinner_activity.completed.is_set()
                assert not window.loading_row.visible
                assert window.open_dataset.enabled
                assert window.path_text.text == str((tmp_path / f"dataset-{run}.csv").resolve())
                assert [row.cells for row in window.dataset_grid.rows] == [(1.5,)]
            finally:
                release_read.set()
                release_conversion.set()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                if read_started.is_set():
                    assert await asyncio.to_thread(read_finished.wait, 5)
                if conversion_started.is_set():
                    assert await asyncio.to_thread(conversion_finished.wait, 5)
                if spinner_activity.started.is_set():
                    assert await asyncio.to_thread(spinner_activity.completed.wait, 5), "The spinner task did not stop."

    asyncio.run(exercise())
    show_error_dialog.assert_not_called()
