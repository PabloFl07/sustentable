"""Exercise real stepper events and confirmed data without requiring a display server."""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from threading import Event as ThreadEvent
from threading import get_ident
from unittest.mock import AsyncMock, Mock

import pytest
from pandas import DataFrame
from pysual import ClickEvent
from pysual.events import Dispatcher

from sustentable.gui import split_step
from sustentable.gui.dataset_window import DatasetWindow
from sustentable.gui.error_dialog import ErrorDialog


class WorkflowWindow(DatasetWindow):
    """Attach Pysual's real dispatcher while leaving native presentation closed."""

    def build(self) -> None:
        """Build the production controls and enable their normal event delivery."""
        super().build()
        self._attach_dispatcher(Dispatcher())

    async def settle(self) -> None:
        """Wait for nested control events and fail on callback errors."""
        async with asyncio.timeout(5):
            await self._dispatcher.drain()

    def ui_thread_id(self) -> int:
        """Identify the UI owner so worker regressions can distinguish its thread."""
        return get_ident()

    async def choose_columns(self, inputs: tuple[int, ...] = (0,), output: int = 2) -> None:
        """Select columns through the actual radio and checkbox controls."""
        self.columns_step._outputs[output].checked = True
        for index, checkbox in enumerate(self.columns_step._inputs):
            checkbox.checked = index in inputs
        await self.settle()

    async def press_continue(self) -> None:
        """Dispatch the common footer action as a click."""
        self.advance.click.emit(ClickEvent(source=self.advance))
        await self.settle()

    async def request_step(self, index: int) -> None:
        """Exercise the same validation path as a header click or arrow key."""
        self.wizard._request_select(index)
        await self.settle()


@pytest.fixture
def window() -> Iterator[WorkflowWindow]:
    """Dispose every control after each workflow scenario."""
    window = WorkflowWindow()
    window.build()
    yield window
    window.destroy()


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    """Create enough rows to exercise the existing default 80/20 split."""
    path = tmp_path / "energy.csv"
    DataFrame({"area": range(10), "occupants": range(10, 20), "energy": range(20, 30)}).to_csv(path, index=False)
    return path


async def _confirmed_split(window: WorkflowWindow, path: Path) -> None:
    await window.show_dataset(path)
    await window.press_continue()
    await window.choose_columns()
    await window.press_continue()
    await window.press_continue()
    assert window.wizard.selected_index == 3


def test_guidance_confirmation_optional_steps_and_shared_preview(window: WorkflowWindow, dataset_path: Path) -> None:
    """Required confirmations gate navigation, optional placeholders can be skipped, and the raw grid persists."""

    async def exercise() -> None:
        assert window.wizard.orientation == "horizontal" and window.wizard.linear
        assert not window.advance.enabled
        await window.request_step(5)
        assert window.wizard.selected_index == 0
        await window.show_dataset(dataset_path)
        await window.press_continue()
        assert window.wizard.selected_index == 1
        assert "output" in window.feedback.text
        assert not window.advance.enabled
        await window.choose_columns()
        assert window.advance.enabled and window.advance.text == "Confirm columns"
        await window.request_step(5)
        assert window.wizard.selected_index == 1
        assert window.selection is None
        await window.press_continue()
        assert window.selection == (["area"], "energy")
        assert window.wizard.selected_index == 2
        assert "Training: 8 rows" in window.split_step.counts.text
        await window.request_step(4)
        assert window.wizard.selected_index == 2
        await window.press_continue()
        assert len(window.train_data) == 8 and len(window.test_data) == 2
        assert sorted([*window.train_data.area, *window.test_data.area]) == list(range(10))
        raw_rows = window.dataset_grid.rows
        for step in window.wizard.steps[3:]:
            assert step.optional and not step.children and not step.completed
        await window.request_step(5)
        assert window.wizard.selected_index == 5
        assert not window.advance.enabled
        assert window.dataset_grid.visible and window.dataset_grid.rows == raw_rows
        window.back.click.emit(ClickEvent(source=window.back))
        await window.settle()
        assert window.wizard.selected_index == 4

    asyncio.run(exercise())


def test_back_preserves_results_and_column_edits_invalidate_dependents(window: WorkflowWindow, dataset_path: Path) -> None:
    """Returning is harmless; changing X/Y invalidates the confirmed selection and both subsets."""

    async def exercise() -> None:
        await _confirmed_split(window, dataset_path)
        train, test = window.train_data, window.test_data
        await window.request_step(1)
        assert window.train_data is train and window.test_data is test
        assert window.columns_step.input_columns == ["area"]
        await window.choose_columns((0, 1))
        assert window.selection is None and window.train_data is None and window.test_data is None
        assert not window.wizard.can_select(2)
        assert window.columns_step.mode.text == "Multiple regression"
        await window.press_continue()
        assert window.selection == (["area", "occupants"], "energy")
        await window.press_continue()
        assert window.train_data.equals(train) and window.test_data.equals(test)

    asyncio.run(exercise())


@pytest.mark.parametrize("invalid_percentage", [0, 1, 100])
def test_split_edits_clear_results_and_empty_subsets_block_confirmation(
    window: WorkflowWindow, dataset_path: Path, invalid_percentage: int
) -> None:
    """Live guidance catches unusable splits and confirmed subsets always match the latest percentage."""

    async def exercise() -> None:
        await _confirmed_split(window, dataset_path)
        await window.request_step(2)
        window.split_step.share.value = invalid_percentage
        await window.settle()
        assert window.train_data is None and window.test_data is None
        assert not window.advance.enabled
        assert "both training and test" in window.feedback.text
        await window.request_step(5)
        assert window.wizard.selected_index == 2
        window.split_step.share.value = 40
        await window.settle()
        assert window.advance.enabled
        await window.press_continue()
        assert len(window.train_data) == 6 and len(window.test_data) == 4

    asyncio.run(exercise())


def test_replacement_resets_workflow_but_failed_load_keeps_confirmed_results(
    window: WorkflowWindow, dataset_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a successful import clears the choices, split and workflow position."""
    errors = Mock()
    monkeypatch.setattr(ErrorDialog, "show_error", errors)

    async def exercise() -> None:
        await _confirmed_split(window, dataset_path)
        train, test, selected = window.train_data, window.test_data, window.selection
        await window.show_dataset(tmp_path / "missing.csv")
        assert errors.call_count == 1
        assert window.wizard.selected_index == 3
        assert window.train_data is train and window.test_data is test and window.selection == selected
        assert window.wizard.can_select(5)
        await window.show_dataset(dataset_path)
        await window.settle()
        assert window.wizard.selected_index == 0
        assert window.selection is None and window.train_data is None and window.test_data is None
        assert not window.columns_step.input_columns and window.columns_step.output_column is None
        assert window.wizard.can_select(1) and not window.wizard.can_select(2)

    asyncio.run(exercise())


def test_output_exclusion_and_duplicate_labels_remain_supported(window: WorkflowWindow) -> None:
    """Column positions still get distinct labels and the selected Y cannot also be an X."""

    async def exercise() -> None:
        window.columns_step.load_columns(["value", "value", 42])
        await window.choose_columns((0, 1), output=1)
        assert [control.text for control in window.columns_step._inputs] == ["value (#1)", "value (#2)", "42"]
        assert window.columns_step.input_columns == ["value"]
        assert not window.columns_step._inputs[1].enabled
        assert not window.columns_step._inputs[1].checked
        assert window.columns_step.output_chip_texts == ["value (#2)"]
        assert window.columns_step.input_chip_texts == ["value (#1)"]

    asyncio.run(exercise())


def test_slider_preview_does_not_shuffle_and_confirmation_runs_in_worker(
    window: WorkflowWindow, dataset_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dragging only previews counts; confirmation shuffles once outside the UI thread."""
    sample = DataFrame.sample
    sampling_threads = []

    def observe_sample(frame: DataFrame, *args: object, **kwargs: object) -> DataFrame:
        sampling_threads.append(get_ident())
        return sample(frame, *args, **kwargs)

    monkeypatch.setattr(DataFrame, "sample", observe_sample)

    async def exercise() -> None:
        await window.show_dataset(dataset_path)
        await window.press_continue()
        await window.choose_columns()
        await window.press_continue()
        for percentage in range(1, 100):
            window.split_step.share.value = percentage
            await window.settle()
        window.split_step.share.value = 40
        await window.settle()
        assert sampling_threads == [], "The preview must never shuffle or copy the dataset."
        assert window.split_step.counts.text == "Training: 6 rows · Test: 4 rows"
        await window.press_continue()
        assert len(sampling_threads) == 1
        assert sampling_threads[0] != window.ui_thread_id()
        assert len(window.train_data) == 6 and len(window.test_data) == 4

    asyncio.run(exercise())


def test_browse_uses_the_step_handler_and_window_request_handler(
    window: WorkflowWindow, dataset_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The button stays owned by the step, and its semantic request reaches the window automatically."""
    chooser = AsyncMock(return_value=dataset_path)
    monkeypatch.setattr(DatasetWindow, "choose_path", chooser)

    async def exercise() -> None:
        assert window.check_handlers(include_unbound=True) == ()
        window.dataset_step.open_dataset.click.emit(ClickEvent(source=window.dataset_step.open_dataset))
        await window.settle()
        chooser.assert_awaited_once()
        assert len(window.dataset) == 10
        assert window.dataset_step.path_text.text == str(dataset_path.resolve())

    asyncio.run(exercise())


@pytest.mark.parametrize("fail", [False, True])
def test_pending_split_stays_responsive_ignores_duplicate_clicks_and_recovers(
    window: WorkflowWindow, dataset_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """A slow or failed split never blocks UI updates or accepts a second confirmation."""
    started, release, finished = ThreadEvent(), ThreadEvent(), ThreadEvent()
    split = split_step.data_shuffler
    calls = []
    errors = Mock()
    monkeypatch.setattr(ErrorDialog, "show_error", errors)

    def blocked_split(*args: object, **kwargs: object) -> tuple[DataFrame, DataFrame]:
        calls.append(get_ident())
        started.set()
        try:
            assert release.wait(5), "The test did not release the split worker."
            if fail:
                raise ValueError("Cannot split these rows.")
            return split(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(split_step, "data_shuffler", blocked_split)

    async def exercise() -> None:
        await window.show_dataset(dataset_path)
        await window.press_continue()
        await window.choose_columns()
        await window.press_continue()
        pending = asyncio.create_task(window.press_continue())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            assert not window.wizard.enabled and not window.advance.enabled and not window.back.enabled
            assert window.loading_row.visible and window.loading_label.text == "Splitting dataset…"
            phase = window.loading_spinner.phase
            async with asyncio.timeout(5):
                while window.loading_spinner.phase == phase:
                    await asyncio.sleep(0.01)
            window.preview_label.text = "The UI remains responsive"
            assert window.preview_label.text == "The UI remains responsive"
            await window.advance_on_click(ClickEvent(source=window.advance))
            assert len(calls) == 1 and calls[0] != window.ui_thread_id()
            assert window.train_data is None and window.test_data is None
        finally:
            release.set()
            await pending
            assert await asyncio.to_thread(finished.wait, 5)
        assert window.wizard.enabled and not window.loading_row.visible
        if fail:
            errors.assert_called_once_with("Cannot split these rows.")
            assert window.wizard.selected_index == 2 and window.advance.enabled
            assert window.train_data is None and window.test_data is None
            monkeypatch.setattr(split_step, "data_shuffler", split)
            await window.press_continue()
        else:
            errors.assert_not_called()
        assert window.wizard.selected_index == 3
        assert len(window.train_data) == 8 and len(window.test_data) == 2

    asyncio.run(exercise())


def test_cancelled_split_discards_late_results_and_allows_retry(
    window: WorkflowWindow, dataset_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling restores navigation, and an abandoned worker cannot overwrite the next confirmed split."""
    started, release, finished = ThreadEvent(), ThreadEvent(), ThreadEvent()
    split = split_step.data_shuffler

    def blocked_split(*args: object, **kwargs: object) -> tuple[DataFrame, DataFrame]:
        started.set()
        try:
            assert release.wait(5)
            return split(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(split_step, "data_shuffler", blocked_split)

    async def exercise() -> None:
        await window.show_dataset(dataset_path)
        await window.press_continue()
        await window.choose_columns()
        await window.press_continue()
        pending = asyncio.create_task(window.advance_on_click(ClickEvent(source=window.advance)))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            async with asyncio.timeout(5):
                while window.loading_row.visible:
                    await asyncio.sleep(0.01)
            assert window.train_data is None and window.test_data is None
            assert window.wizard.enabled and window.advance.enabled
            monkeypatch.setattr(split_step, "data_shuffler", split)
            window.split_step.share.value = 40
            await window.settle()
            await window.press_continue()
            assert len(window.train_data) == 6 and len(window.test_data) == 4
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            assert await asyncio.to_thread(finished.wait, 5)
        assert len(window.train_data) == 6 and len(window.test_data) == 4

    asyncio.run(exercise())
