"""Tests for the start screen: the window opens on it, and its two choices open the dataset or the model dialog.

File dialogs and error presentation are mocked; tile clicks travel through Pysual's real event dispatcher.
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock, Mock

import pytest
from pandas import DataFrame
from pysual import CapabilityError, ClickEvent, UiEvent
from pysual.events import Dispatcher

from sustentable.core.io.readers import SUPPORTED_EXTENSIONS
from sustentable.gui.controls.action_tile import ActionTile
from sustentable.gui.controls.error_dialog import ErrorDialog
from sustentable.gui.dataset_window import OPEN_ERROR_TITLE, DatasetWindow


class StartWindow(DatasetWindow):
    """Attach Pysual's real dispatcher while leaving native presentation closed."""

    def build(self) -> None:
        """Build the production controls and enable their normal event delivery."""
        super().build()
        self._attach_dispatcher(Dispatcher())

    async def click(self, tile: ActionTile) -> None:
        """Click ``tile`` and wait for every handler the click triggers."""
        tile.click.emit(ClickEvent(source=tile))
        async with asyncio.timeout(5):
            await self._dispatcher.drain()


@pytest.fixture
def window() -> Iterator[StartWindow]:
    """Dispose every control after each scenario."""
    window = StartWindow()
    window.build()
    yield window
    window.destroy()


@pytest.fixture
def errors(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Capture the window's requests to its error dialog."""
    errors = Mock()
    monkeypatch.setattr(ErrorDialog, "show_error", errors)
    return errors


def _showing_start(window: StartWindow) -> bool:
    """Whether the start dialog is open over the workflow and no data is shown yet."""
    hidden = (window.preview, window.fit_panel, window.no_data_panel)
    return (
        window.start_screen.visible
        and window.wizard.visible
        and window.footer.visible
        and not window.load_model_button.visible
        and not any(control.visible for control in hidden)
    )


def test_the_window_opens_on_the_start_screen(window: StartWindow) -> None:
    """At startup the start dialog offers both choices, and the workflow is already behind it."""
    screen = window.start_screen

    assert _showing_start(window)
    assert (screen.model_tile.text, screen.dataset_tile.text) == ("Load model", "Load dataset")
    assert screen.model_tile.effective_enabled and screen.dataset_tile.effective_enabled
    assert window.check_handlers(include_unbound=True) == ()


def test_load_dataset_opens_the_dataset_dialog_and_shows_the_workflow(
    window: StartWindow, example_csv: Path, monkeypatch: pytest.MonkeyPatch, errors: Mock
) -> None:
    """The chosen file loads into the dataset step, and the start dialog closes."""
    chooser = AsyncMock(return_value=example_csv)
    monkeypatch.setattr(DatasetWindow, "choose_path", chooser)

    asyncio.run(window.click(window.start_screen.dataset_tile))

    chooser.assert_awaited_once_with(title="Open dataset", extensions=SUPPORTED_EXTENSIONS)
    errors.assert_not_called()
    assert not window.start_screen.visible
    assert window.wizard.visible and window.preview.visible and window.footer.visible and window.feedback.visible
    assert not window.fit_panel.visible
    assert window.wizard.selected_step is window.dataset_step and not window.advance.enabled
    assert "output" in window.feedback.text
    assert window.dataset_step.path_text.text == str(example_csv.resolve())
    assert [row.cells for row in window.dataset_grid.rows] == [("José", 19), ("Begoña", 21)]


@pytest.mark.parametrize("outcome", ["cancelled", "dialog error", "missing file", "damaged file"])
def test_cancelled_or_failed_dataset_keeps_the_start_screen_and_allows_another_try(
    window: StartWindow, example_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, errors: Mock, outcome: str
) -> None:
    """Closing the dialog changes nothing silently, a failure is reported, and a later valid file still loads."""
    damaged = tmp_path / "damaged.csv"
    damaged.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe" * 20)
    first = {
        "cancelled": None,
        "dialog error": CapabilityError("File dialog unavailable."),
        "missing file": tmp_path / "missing.csv",
        "damaged file": damaged,
    }[outcome]
    monkeypatch.setattr(DatasetWindow, "choose_path", AsyncMock(side_effect=[first, example_csv]))

    async def exercise() -> None:
        await window.click(window.start_screen.dataset_tile)
        assert _showing_start(window) and window.dataset is None
        if outcome == "cancelled":
            errors.assert_not_called()
        else:
            errors.assert_called_once()
            assert errors.call_args.kwargs == {"title": OPEN_ERROR_TITLE}
        assert window.start_screen.dataset_tile.effective_enabled

        await window.click(window.start_screen.dataset_tile)
        assert not window.start_screen.visible and window.wizard.visible
        assert window.dataset_step.path_text.text == str(example_csv.resolve())

    asyncio.run(exercise())


def test_load_model_opens_the_model_dialog_and_cancelling_keeps_the_start_screen(
    window: StartWindow, monkeypatch: pytest.MonkeyPatch, errors: Mock
) -> None:
    """The dialog only offers saved models, and closing it without choosing one is not an error."""
    chooser = AsyncMock(return_value=None)
    monkeypatch.setattr(DatasetWindow, "choose_path", chooser)

    asyncio.run(window.click(window.start_screen.model_tile))

    chooser.assert_awaited_once_with(title="Open model", extensions=(".onnx",))
    errors.assert_not_called()
    assert _showing_start(window)


def test_both_choices_are_disabled_while_a_dataset_loads(
    window: StartWindow, example_csv: Path, monkeypatch: pytest.MonkeyPatch, errors: Mock
) -> None:
    """The loading message appears below the choices, and neither choice reacts until the dataset has loaded."""
    started, release = Event(), Event()
    frame = DataFrame({"score": [1.5]})

    def blocked_read(path: str | Path) -> tuple[str, DataFrame]:
        started.set()
        assert release.wait(5), "The test did not release the dataset reader."
        return str(Path(path).resolve()), frame

    monkeypatch.setattr("sustentable.gui.dataset_window._read_dataset", blocked_read)
    chooser = AsyncMock(return_value=example_csv)
    monkeypatch.setattr(DatasetWindow, "choose_path", chooser)
    screen = window.start_screen

    async def exercise() -> None:
        pending = asyncio.create_task(window.click(screen.dataset_tile))
        try:
            assert await asyncio.to_thread(started.wait, 5), "The dataset reader did not start."
            assert _showing_start(window) and window.loading_row.visible
            assert window.loading_label.text == "Loading dataset…"
            assert not screen.model_tile.effective_enabled and not screen.dataset_tile.effective_enabled
            await window.start_screen_on_load_model(UiEvent(source=screen))
            await window.start_screen_on_load_dataset(UiEvent(source=screen))
            assert chooser.await_count == 1
        finally:
            release.set()
            await pending

    asyncio.run(exercise())

    errors.assert_not_called()
    assert not window.loading_row.visible
    assert not screen.visible and window.dataset is frame
