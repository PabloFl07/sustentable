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

from sustentable.core.processing.data_processing import NanStrategy, detect_nan_values, has_nan_values
from sustentable.gui import dataset_window
from sustentable.gui.controls.error_dialog import ErrorDialog
from sustentable.gui.dataset_window import MODEL_ERROR_TITLE, DatasetWindow
from sustentable.gui.steps import split_step
from sustentable.gui.steps.preprocess_step import STRATEGY_LABELS


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

    async def create_model(self) -> None:
        """Dispatch the model step's button as a click."""
        button = self.model_step.create_button
        button.click.emit(ClickEvent(source=button))
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
        assert window.preprocess_step.completed and not window.preprocess_step.has_missing_values
        await window.request_step(5)
        assert window.wizard.selected_index == 4, "Predictions need a model first."
        await window.create_model()
        for step in window.wizard.steps[5:]:
            assert step.optional and not step.children and not step.completed
        await window.request_step(5)
        assert window.wizard.selected_index == 5
        assert not window.advance.enabled
        assert window.preview.visible and window.dataset_grid.visible and window.dataset_grid.rows == raw_rows
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


@pytest.mark.parametrize("invalid_percentage", [0, 99, 100])
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
    """Only a successful import clears the choices, split, model and workflow position."""
    errors = Mock()
    monkeypatch.setattr(ErrorDialog, "show_error", errors)

    async def exercise() -> None:
        await _confirmed_split(window, dataset_path)
        await window.create_model()
        train, test, selected, model = window.train_data, window.test_data, window.selection, window.model
        await window.show_dataset(tmp_path / "missing.csv")
        assert errors.call_count == 1
        assert window.wizard.selected_index == 3
        assert window.train_data is train and window.test_data is test and window.selection == selected
        assert window.model is model and model is not None
        assert window.wizard.can_select(5)
        await window.show_dataset(dataset_path)
        await window.settle()
        assert window.wizard.selected_index == 0
        assert window.selection is None and window.train_data is None and window.test_data is None
        assert window.model is None and not window.fit_chart.point_series
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
    """Dragging only previews counts; confirmation splits once outside the UI thread."""
    split = split_step.split_dataset
    sampling_threads = []

    def observe_split(*args: object, **kwargs: object) -> tuple[DataFrame, DataFrame]:
        sampling_threads.append(get_ident())
        return split(*args, **kwargs)

    monkeypatch.setattr(split_step, "split_dataset", observe_split)

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
    split = split_step.split_dataset
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

    monkeypatch.setattr(split_step, "split_dataset", blocked_split)

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
            errors.assert_called_once_with("Cannot split these rows.", title="Unable to split dataset")
            assert window.wizard.selected_index == 2 and window.advance.enabled
            assert window.train_data is None and window.test_data is None
            monkeypatch.setattr(split_step, "split_dataset", split)
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
    split = split_step.split_dataset

    def blocked_split(*args: object, **kwargs: object) -> tuple[DataFrame, DataFrame]:
        started.set()
        try:
            assert release.wait(5)
            return split(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(split_step, "split_dataset", blocked_split)

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
            monkeypatch.setattr(split_step, "split_dataset", split)
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


@pytest.fixture
def missing_path(tmp_path: Path) -> Path:
    """Create a dataset with missing values in the input column and in an unselected column."""
    path = tmp_path / "missing.csv"
    area = [None if index in (1, 4, 7) else index for index in range(10)]
    DataFrame({"area": area, "occupants": [None] * 10, "energy": range(20, 30)}).to_csv(path, index=False)
    return path


async def _apply(window: WorkflowWindow, strategy: NanStrategy, constant: str = "") -> None:
    window.preprocess_step.strategy.selected_index = list(STRATEGY_LABELS).index(strategy)
    window.preprocess_step.constant.text = constant
    await window.settle()
    window.preprocess_step.apply_button.click.emit(ClickEvent(source=window.preprocess_step.apply_button))
    await window.settle()


def test_missing_values_block_the_model_until_preprocessing_is_applied(window: WorkflowWindow, missing_path: Path) -> None:
    """Missing values are reported per subset, and each strategy must be applied before continuing."""

    async def exercise() -> None:
        await _confirmed_split(window, missing_path)
        step = window.preprocess_step
        report = detect_nan_values(window.train_data, window.test_data, ["area", "energy"])
        assert step.has_missing_values and report.train.sum() + report.test.sum() == 3
        assert [row.text for row in step._rows][0] == f"area: training {report.train[0]} · test {report.test[0]}"
        assert not window.advance.enabled and "Apply preprocessing" in window.feedback.text
        await window.request_step(4)
        assert window.wizard.selected_index == 3
        assert not step.constant.enabled

        await _apply(window, NanStrategy.MEAN)
        train, test = window.preprocessed_data
        assert not has_nan_values(train, test, ["area", "energy"])
        assert test.area.isin([window.train_data.area.mean()]).sum() == report.test[0]
        assert "values filled" in step.status.text and window.advance.enabled
        assert window.train_data.area.isna().sum() == report.train[0], "The split itself is not modified."

        await _apply(window, NanStrategy.DELETE)
        train, test = window.preprocessed_data
        assert len(train) + len(test) == 7 and "rows removed" in step.status.text

        window.preprocess_step.strategy.selected_index = list(STRATEGY_LABELS).index(NanStrategy.CONSTANT)
        await window.settle()
        assert step.constant.enabled and window.preprocessed_data is None and not window.advance.enabled
        await _apply(window, NanStrategy.CONSTANT, "-1")
        assert window.preprocessed_data[0].area.min() == -1
        await window.press_continue()
        assert window.wizard.selected_index == 4

    asyncio.run(exercise())


def test_preprocessing_errors_keep_the_step_retryable(window: WorkflowWindow, missing_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric constant shows an error without applying anything, and the user can retry."""
    errors = Mock()
    monkeypatch.setattr(ErrorDialog, "show_error", errors)

    async def exercise() -> None:
        await _confirmed_split(window, missing_path)
        await _apply(window, NanStrategy.CONSTANT, "abc")
        errors.assert_called_once()
        assert "number" in errors.call_args.args[0]
        assert window.preprocessed_data is None and not window.advance.enabled
        await _apply(window, NanStrategy.CONSTANT, "0")
        assert window.preprocessed_data is not None and window.advance.enabled

    asyncio.run(exercise())


def test_split_and_column_changes_require_preprocessing_again(window: WorkflowWindow, missing_path: Path) -> None:
    """Repeating the split or changing the columns discards the applied preprocessing."""

    async def exercise() -> None:
        await _confirmed_split(window, missing_path)
        await _apply(window, NanStrategy.MEDIAN)
        assert window.preprocessed_data is not None
        await window.request_step(2)
        window.split_step.share.value = 30
        await window.settle()
        assert window.preprocessed_data is None and not window.preprocess_step.has_missing_values
        await window.press_continue()
        assert window.wizard.selected_index == 3 and window.preprocessed_data is None
        await _apply(window, NanStrategy.MEDIAN)
        assert window.preprocessed_data is not None
        await window.request_step(1)
        await window.choose_columns((0, 1))
        assert window.preprocessed_data is None and not window.wizard.can_select(3)

    asyncio.run(exercise())


def test_applied_preprocessing_reloads_the_preview_and_the_counters(window: WorkflowWindow, missing_path: Path) -> None:
    """The table and counters show the processed subsets, and return to the dataset when the result is discarded."""

    async def exercise() -> None:
        await _confirmed_split(window, missing_path)
        step = window.preprocess_step
        raw_rows, raw_caption = window.dataset_grid.rows, window.preview_label.text
        assert step.remaining_missing_values != (0, 0)
        original = detect_nan_values(window.train_data, window.test_data, ["area", "energy"])

        await _apply(window, NanStrategy.DELETE)
        train, test = window.preprocessed_data
        assert len(window.dataset_grid.rows) == len(train) + len(test) == 7
        assert window.preview_label.text == "missing.csv · preprocessed · training " + f"{len(train)} + test {len(test)} rows"
        assert step.remaining_missing_values == (0, 0)
        assert step.summary.text == "Missing values after preprocessing: training 0 · test 0"
        assert step._rows[0].text.startswith(f"area: training 0 (was {original.train[0]})")
        assert step._rows[1].text == "energy: training 0 · test 0"

        await _apply(window, NanStrategy.MEAN)
        assert len(window.dataset_grid.rows) == 10 and step.remaining_missing_values == (0, 0)

        window.preprocess_step.strategy.selected_index = list(STRATEGY_LABELS).index(NanStrategy.MEDIAN)
        await window.settle()
        assert window.dataset_grid.rows == raw_rows and window.preview_label.text == raw_caption
        assert step.remaining_missing_values == (int(original.train.sum()), int(original.test.sum()))
        assert "(was" not in step._rows[0].text

        await _apply(window, NanStrategy.MEDIAN)
        await window.request_step(1)
        await window.choose_columns((0,))
        await window.choose_columns((0,), output=1)
        assert window.dataset_grid.rows == raw_rows and window.preview_label.text == raw_caption

    asyncio.run(exercise())


def test_failed_preprocessing_restores_the_dataset_preview(window: WorkflowWindow, missing_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An error after a successful apply discards it and shows the loaded dataset again."""
    monkeypatch.setattr(ErrorDialog, "show_error", Mock())

    async def exercise() -> None:
        await _confirmed_split(window, missing_path)
        raw_rows = window.dataset_grid.rows
        await _apply(window, NanStrategy.DELETE)
        assert len(window.dataset_grid.rows) == 7
        await _apply(window, NanStrategy.CONSTANT, "abc")
        assert window.preprocessed_data is None and window.dataset_grid.rows == raw_rows
        assert "(was" not in window.preprocess_step._rows[0].text

    asyncio.run(exercise())


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Create rows where energy = 2 * area - 3 * occupants + 50 exactly, plus a text column."""
    path = tmp_path / "model.csv"
    area = [3, 7, 1, 9, 4, 8, 2, 6, 5, 0]
    occupants = [1, 4, 2, 2, 5, 3, 0, 1, 4, 3]
    energy = [2 * a - 3 * o + 50 for a, o in zip(area, occupants, strict=True)]
    DataFrame({"area": area, "occupants": occupants, "energy": energy, "city": "Lugo"}).to_csv(path, index=False)
    return path


async def _open_model_step(window: WorkflowWindow, path: Path, inputs: tuple[int, ...]) -> None:
    """Load a dataset without missing values, confirm the columns and the default split, and open the model step."""
    await window.show_dataset(path)
    await window.press_continue()
    await window.choose_columns(inputs)
    for _ in range(3):
        await window.press_continue()
    assert window.wizard.selected_index == 4


def test_create_model_needs_confirmed_columns_split_and_no_missing_values(window: WorkflowWindow, missing_path: Path) -> None:
    """The button explains each missing prerequisite and is only enabled once all of them are met."""

    async def exercise() -> None:
        step, button = window.model_step, window.model_step.create_button
        assert not button.enabled and "columns" in step.status.text
        await window.show_dataset(missing_path)
        await window.press_continue()
        await window.choose_columns()
        assert not button.enabled and "columns" in step.status.text
        await window.press_continue()
        assert not button.enabled and "split" in step.status.text
        await window.press_continue()
        assert not button.enabled and "missing values" in step.status.text
        await window.create_model()
        assert window.model is None

        await _apply(window, NanStrategy.MEAN)
        assert button.enabled and step.status.text.startswith("Ready")
        await window.press_continue()
        await window.create_model()
        train, test = window.preprocessed_data
        assert (window.model.train_metrics.rows, window.model.test_metrics.rows) == (len(train), len(test))

        await window.request_step(3)
        await _apply(window, NanStrategy.DELETE)
        assert window.model is None, "A new treatment changes the data, so the model must be created again."
        assert button.enabled and not step.completed

    asyncio.run(exercise())


def test_simple_model_shows_its_formula_metrics_and_fit(window: WorkflowWindow, dataset_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The model is trained in a worker with the confirmed subsets, and the fit chart replaces the table on its step."""
    calls = []
    create = dataset_window.create_model

    def observe(*args: object) -> object:
        calls.append((get_ident(), args))
        return create(*args)

    monkeypatch.setattr(dataset_window, "create_model", observe)

    async def exercise() -> None:
        await _open_model_step(window, dataset_path, (0,))
        assert window.model is None and not window.advance.enabled and window.advance.text == "Continue"
        assert window.feedback.text == "Create the model to continue."
        assert window.preview.visible and not window.fit_panel.visible

        await window.create_model()
        model = window.model
        [(thread, (train, test, inputs, output))] = calls
        assert thread != window.ui_thread_id()
        assert train is window.preprocessed_data[0] and test is window.preprocessed_data[1]
        assert (inputs, output) == (["area"], "energy")
        step = window.model_step
        assert step.formula.text == model.formula == "energy = 1.0000 * area + 20.0000"
        assert [label.text for label in step.r2_values] == ["1.0000", "1.0000"]
        assert [label.text for label in step.mse_values] == ["0.0000", "0.0000"]
        assert step.status.text == "Model created successfully. Simple regression trained on 8 rows and tested on 2 rows."
        assert window.feedback.text == step.status.text
        assert step.completed and window.advance.enabled

        assert window.fit_panel.visible and not window.preview.visible
        chart = window.fit_chart
        assert (chart.x_label, chart.y_label) == ("area", "energy")
        training, testing = chart.point_series
        assert (training.title, testing.title) == ("Training", "Test")
        assert (training.shape, testing.shape) == ("circle", "square") and training.color != testing.color
        assert sorted(training.points) == sorted(zip(train.area.astype(float), train.energy.astype(float), strict=True))
        assert len(testing.points) == 2
        (line,) = chart.line_series
        assert line.title == "Regression line"
        assert line.start == pytest.approx((0.0, 20.0)) and line.end == pytest.approx((9.0, 29.0))
        assert model.predict(DataFrame({"area": [100]})) == pytest.approx([120.0])

        await window.request_step(3)
        assert window.preview.visible and not window.fit_panel.visible
        await window.request_step(5)
        assert window.wizard.selected_index == 5 and window.model is model

    asyncio.run(exercise())


def test_new_model_replaces_the_previous_one_and_data_changes_discard_it(window: WorkflowWindow, model_path: Path) -> None:
    """Creating the model again replaces the results; changing the columns or the split clears them."""

    async def exercise() -> None:
        await _open_model_step(window, model_path, (0,))
        await window.create_model()
        first = window.model
        step = window.model_step
        assert first.is_simple and step.formula_caption.text == "FORMULA · SIMPLE REGRESSION"
        await window.create_model()
        assert window.model is not first and window.model.formula == first.formula

        await window.request_step(1)
        await window.choose_columns((0, 1))
        assert window.model is None and not window.fit_chart.point_series and not window.fit_panel.visible
        assert step.formula.text == "Create the model to see its formula."
        assert [label.text for label in (*step.r2_values, *step.mse_values)] == ["—"] * 4
        assert not step.completed and not step.create_button.enabled

        for _ in range(3):
            await window.press_continue()
        await window.create_model()
        assert step.formula.text == "energy = 2.0000 * area - 3.0000 * occupants + 50.0000"
        assert step.formula_caption.text == "FORMULA · MULTIPLE REGRESSION"
        assert [label.text for label in step.r2_values] == ["1.0000", "1.0000"]
        chart = window.fit_chart
        assert (chart.x_label, chart.y_label) == ("Actual value (energy)", "Predicted value (energy)")
        assert [line.title for line in chart.line_series] == ["Predicted = actual"]
        training, testing = chart.point_series
        assert (len(training.points), len(testing.points)) == (8, 2)
        for actual, predicted in (*training.points, *testing.points):
            assert predicted == pytest.approx(actual)

        await window.request_step(2)
        window.split_step.share.value = 30
        await window.settle()
        assert window.model is None and not window.fit_chart.point_series

    asyncio.run(exercise())


def test_model_errors_are_reported_and_the_window_keeps_working(
    window: WorkflowWindow, model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A text input or a failed fit shows an explanatory error, keeps any previous model and allows a retry."""
    errors = Mock()
    monkeypatch.setattr(ErrorDialog, "show_error", errors)

    async def exercise() -> None:
        await _open_model_step(window, model_path, (3,))
        await window.create_model()
        errors.assert_called_once()
        assert "'city' is not numeric" in errors.call_args.args[0]
        assert errors.call_args.kwargs == {"title": MODEL_ERROR_TITLE}
        assert window.model is None and not window.fit_panel.visible
        assert window.wizard.enabled and not window.loading_row.visible and window.model_step.create_button.enabled

        await window.request_step(1)
        await window.choose_columns((0,))
        for _ in range(3):
            await window.press_continue()
        await window.create_model()
        previous, points = window.model, window.fit_chart.point_series
        assert previous is not None

        def fail(*args: object) -> None:
            raise ValueError("The model could not be fitted: singular matrix.")

        monkeypatch.setattr(dataset_window, "create_model", fail)
        await window.create_model()
        errors.assert_called_with("The model could not be fitted: singular matrix.", title=MODEL_ERROR_TITLE)
        assert window.model is previous and window.fit_chart.point_series is points and window.fit_panel.visible
        assert window.model_step.formula.text == previous.formula
        assert window.wizard.enabled and window.model_step.create_button.enabled

    asyncio.run(exercise())
