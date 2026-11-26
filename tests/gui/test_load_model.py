"""Load a saved model from the GUI: the model step without data, editing and saving it again, and replacing it.

Models are trained and saved in each test. File dialogs and message dialogs are mocked; clicks travel through Pysual's
real event dispatcher.
"""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock, Mock

import onnx
import pytest
from pandas import DataFrame
from pysual import Button, CapabilityError, ClickEvent
from pysual.events import Dispatcher

from sustentable.core.io.export_model import export_model
from sustentable.core.io.import_model import SavedModel
from sustentable.core.model.regression import create_model
from sustentable.gui import dataset_window
from sustentable.gui.controls.confirm_dialog import ConfirmDialog
from sustentable.gui.controls.error_dialog import ErrorDialog
from sustentable.gui.dataset_window import LOAD_ERROR_TITLE, LOADED_TITLE, SAVED_TITLE, DatasetWindow

# energy = 2 * area - 3 * occupants + 50 exactly, so the formula and the predictions are known.
AREA = [3, 7, 1, 9, 4, 8, 2, 6, 5, 0]
OCCUPANTS = [1, 4, 2, 2, 5, 3, 0, 1, 4, 3]
DATA = DataFrame({"area": AREA, "occupants": OCCUPANTS, "energy": [2 * a - 3 * o + 50 for a, o in zip(AREA, OCCUPANTS, strict=True)]})
FORMULA = "energy = 2.0000 * area - 3.0000 * occupants + 50.0000"
DATASET, COLUMNS, SPLIT, PREPROCESS, MODEL, PREDICT = range(6)


class ModelWindow(DatasetWindow):
    """Attach Pysual's real dispatcher while leaving native presentation closed."""

    def build(self) -> None:
        """Build the production controls and enable their normal event delivery."""
        super().build()
        self._attach_dispatcher(Dispatcher())

    async def settle(self) -> None:
        """Wait for nested control events and fail on callback errors."""
        async with asyncio.timeout(5):
            await self._dispatcher.drain()

    async def click(self, button: Button) -> None:
        """Click ``button`` and wait for every handler the click triggers."""
        button.click.emit(ClickEvent(source=button))
        await self.settle()

    async def request_step(self, index: int) -> None:
        """Exercise the same validation path as a header click or arrow key."""
        self.wizard._request_select(index)
        await self.settle()

    async def type_description(self, text: str) -> None:
        """Replace the description of the model shown, as typing does."""
        self.model_step.description.text = text
        await self.settle()


@pytest.fixture
def window() -> Iterator[ModelWindow]:
    """Dispose every control after each scenario."""
    window = ModelWindow()
    window.build()
    yield window
    window.destroy()


@pytest.fixture
def dialogs(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, AsyncMock]:
    """Replace the error, notice and confirmation dialogs; every confirmation is accepted unless a test changes it."""
    errors, notices, confirm = Mock(), Mock(), AsyncMock(return_value=True)

    def show(dialog: ErrorDialog, *args: object, **kwargs: object) -> None:
        (notices if kwargs.get("title") in (LOADED_TITLE, SAVED_TITLE) else errors)(*args, **kwargs)

    monkeypatch.setattr(ErrorDialog, "show_error", show)
    monkeypatch.setattr(ConfirmDialog, "ask", confirm)
    return errors, notices, confirm


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A folder whose name has spaces, accents and ñ, as the absolute path the loader reports."""
    folder = tmp_path / "Modelos de energía ñ"
    folder.mkdir()
    return folder.resolve()


def _save_model(path: Path, inputs: list[str], description: str) -> Path:
    """Train the model of ``DATA`` on ``inputs`` and save it to ``path``."""
    model = create_model(DATA.head(8), DATA.tail(2), inputs, "energy")
    return export_model(model, path, description).model


@pytest.fixture
def model_file(folder: Path) -> Path:
    """A saved multiple regression with a description."""
    return _save_model(folder / "consumo energía.onnx", ["area", "occupants"], "Predice la energía.")


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    """A dataset to replace the loaded model with."""
    path = tmp_path / "energy.csv"
    DATA.to_csv(path, index=False)
    return path


def _choose(monkeypatch: pytest.MonkeyPatch, *answers: object) -> AsyncMock:
    """Make the file dialogs return ``answers`` in order."""
    chooser = AsyncMock(side_effect=[str(answer) if isinstance(answer, Path) else answer for answer in answers])
    monkeypatch.setattr(DatasetWindow, "choose_path", chooser)
    return chooser


def _metadata(path: Path) -> dict[str, object]:
    """Every metadata entry of a saved file, decoded."""
    return {prop.key: json.loads(prop.value) for prop in onnx.load(path).metadata_props}


def _showing_loaded_model(window: ModelWindow) -> bool:
    """Whether the steps show a model without data: the data steps are disabled and a note replaces the table and chart."""
    data_steps = (window.columns_step, window.split_step, window.preprocess_step)
    hidden = (window.start_screen, window.preview, window.fit_panel)
    return (
        window.wizard.visible
        and window.footer.visible
        and window.no_data_panel.visible
        and not any(control.visible for control in hidden)
        and not any(step.enabled for step in data_steps)
        and not window.wizard.linear
        and window.dataset is None
    )


async def _load_from_start(window: ModelWindow) -> None:
    await window.click(window.start_screen.model_tile)


async def _open_dataset_step(window: ModelWindow) -> None:
    """Go back from the model step to the dataset step and click its file button."""
    await window.click(window.back)
    assert window.wizard.selected_index == DATASET
    await window.click(window.dataset_step.open_dataset)


# ----------------------------------------------------------------------------
# Loading and showing the model
# ----------------------------------------------------------------------------


def test_loaded_model_opens_in_the_model_step_without_the_data_sections(
    window: ModelWindow, model_file: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """Formula, columns, metrics and description are shown for consultation, with a confirmation message and no chart."""
    errors, notices, confirm = dialogs
    chooser = _choose(monkeypatch, model_file)

    asyncio.run(_load_from_start(window))

    chooser.assert_awaited_once_with(title="Open model", extensions=(".onnx",))
    errors.assert_not_called()
    confirm.assert_not_awaited()
    notices.assert_called_once()
    assert notices.call_args.kwargs == {"title": LOADED_TITLE} and str(model_file) in notices.call_args.args[0]
    assert _showing_loaded_model(window) and window.wizard.selected_step is window.model_step
    step = window.model_step
    assert window.model is step.model is window.predict_step.model is not None
    assert step.formula.text == FORMULA and step.formula_caption.text == "FORMULA · MULTIPLE REGRESSION"
    assert step.columns_text.text == "Output (Y): energy\nInputs (X): area, occupants"
    assert [label.text for label in (*step.r2_values, *step.mse_values)] == ["1.0000", "1.0000", "0.0000", "0.0000"]
    assert step.description.text == "Predice la energía."
    assert step.status.text == "Model loaded from consumo energía.onnx" and step.status.tooltip == str(model_file)
    assert step.columns_text.visible and not step.create_button.visible
    assert step.save_button.enabled and not step.has_unsaved_changes and step.saved_path == model_file
    assert window.feedback.text == step.status.text
    assert not window.fit_chart.point_series and not window.dataset_grid.rows
    assert window.check_handlers(include_unbound=True) == ()


def test_steps_that_work_with_data_are_skipped_while_a_loaded_model_is_shown(
    window: ModelWindow, model_file: Path, dataset_file: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """Only the dataset, model and prediction steps can be entered; a new dataset brings back the guided order."""
    _choose(monkeypatch, model_file, dataset_file)

    async def exercise() -> None:
        await _load_from_start(window)
        for index in (COLUMNS, SPLIT, PREPROCESS):
            await window.request_step(index)
            assert window.wizard.selected_index == MODEL
        assert window.advance.enabled
        await window.click(window.advance)
        assert window.wizard.selected_step is window.predict_step and _showing_loaded_model(window)
        await window.click(window.back)
        assert window.wizard.selected_step is window.model_step
        await window.click(window.back)
        assert window.wizard.selected_step is window.dataset_step, "Back skips the disabled steps."
        assert window.feedback.text == "Open a dataset to create a new model." and not window.advance.enabled
        await window.request_step(MODEL)
        assert window.wizard.selected_step is window.model_step

        await _open_dataset_step(window)
        assert window.dataset is not None and window.model is None and window.wizard.selected_index == DATASET
        assert window.wizard.linear and all(step.enabled for step in window.wizard.steps)
        assert window.preview.visible and not window.no_data_panel.visible
        assert window.model_step.description.text == "", "The description belonged to the loaded model."
        assert not window.model_step.columns_text.visible and window.model_step.create_button.visible
        assert not window.wizard.can_select(MODEL)
        await window.request_step(MODEL)
        assert window.wizard.selected_index == COLUMNS, "The guided order applies again, starting at the first step to do."

    asyncio.run(exercise())


def test_loaded_model_makes_predictions(
    window: ModelWindow, model_file: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """The prediction step only needs the model, so it works without the dataset."""
    _choose(monkeypatch, model_file)

    async def exercise() -> None:
        await _load_from_start(window)
        await window.click(window.advance)
        prediction = window.predict_step
        assert list(prediction.fields) == ["area", "occupants"]
        prediction.fields["area"].text = "10"
        prediction.fields["occupants"].text = "2,5"
        await window.click(prediction.predict_button)
        assert prediction.result.text == "energy = 62.5000" and not prediction.has_error

    asyncio.run(exercise())


def test_model_saved_without_description_shows_an_empty_description(
    window: ModelWindow, folder: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """An empty description is valid, so the text area is empty and the model has no unsaved changes."""
    _choose(monkeypatch, _save_model(folder / "simple.onnx", ["area"], ""))

    asyncio.run(_load_from_start(window))

    step = window.model_step
    assert step.description.text == "" and not step.has_unsaved_changes
    assert step.formula_caption.text == "FORMULA · SIMPLE REGRESSION"
    assert step.columns_text.text == "Output (Y): energy\nInputs (X): area"


@pytest.mark.parametrize("outcome", ["cancelled", "dialog error", "missing file", "damaged file"])
def test_cancelled_or_failed_load_keeps_the_previous_state_and_allows_another_try(
    window: ModelWindow,
    model_file: Path,
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    dialogs: tuple[Mock, Mock, AsyncMock],
    outcome: str,
) -> None:
    """Closing the dialog changes nothing silently; a failure shows a clear error and the user can choose another file."""
    errors, notices, _ = dialogs
    damaged = folder / "dañado.onnx"
    damaged.write_bytes(model_file.read_bytes()[:100])
    first = {
        "cancelled": None,
        "dialog error": CapabilityError("File dialog unavailable."),
        "missing file": folder / "missing.onnx",
        "damaged file": damaged,
    }[outcome]
    other_file = _save_model(folder / "simple.onnx", ["area"], "Otro modelo")
    _choose(monkeypatch, first, model_file, first, other_file)

    async def exercise() -> None:
        await _load_from_start(window)
        assert window.start_screen.visible and window.model is None
        assert errors.call_count == (outcome != "cancelled")
        if errors.called:
            assert errors.call_args.kwargs == {"title": LOAD_ERROR_TITLE}
            if outcome != "dialog error":
                assert str(first) in errors.call_args.args[0]

        await _load_from_start(window)
        loaded = window.model
        assert _showing_loaded_model(window) and loaded is not None

        await window.click(window.load_model_button)
        assert window.model is loaded, "A failed replacement keeps the model shown."
        assert window.model_step.description.text == "Predice la energía."
        await window.click(window.load_model_button)
        assert window.model_step.description.text == "Otro modelo" and window.model is not loaded
        assert notices.call_count == 2

    asyncio.run(exercise())


def test_model_loads_in_a_worker_while_the_choices_are_disabled(
    window: ModelWindow, model_file: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """Reading the file never blocks the window, and no other file can be chosen until it finishes."""
    started, release = Event(), Event()
    load = dataset_window.import_model

    def blocked_load(path: str | Path) -> SavedModel:
        started.set()
        assert release.wait(5), "The test did not release the model loader."
        return load(path)

    monkeypatch.setattr(dataset_window, "import_model", blocked_load)
    chooser = _choose(monkeypatch, model_file)

    async def exercise() -> None:
        pending = asyncio.create_task(_load_from_start(window))
        try:
            assert await asyncio.to_thread(started.wait, 5), "The model loader did not start."
            assert window.loading_row.visible and window.loading_label.text == "Loading model…"
            assert not window.start_screen.model_tile.effective_enabled and not window.start_screen.dataset_tile.effective_enabled
            await window.show_saved_model(model_file)
            assert chooser.await_count == 1
        finally:
            release.set()
            await pending

    asyncio.run(exercise())
    assert not window.loading_row.visible and _showing_loaded_model(window)
    assert window.load_model_button.effective_enabled and window.dataset_step.open_dataset.effective_enabled


# ----------------------------------------------------------------------------
# Editing the description and saving again
# ----------------------------------------------------------------------------


def test_edited_description_is_saved_again_to_the_same_file_keeping_the_rest(
    window: ModelWindow, model_file: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """Only the description can change; saving proposes the loaded file and keeps the coefficients, columns and metrics."""
    errors, notices, _ = dialogs
    before = _metadata(model_file)
    chooser = _choose(monkeypatch, model_file, model_file)

    async def exercise() -> None:
        await _load_from_start(window)
        step = window.model_step
        await window.type_description("Predice la energía de cada vivienda.")
        assert step.has_unsaved_changes
        assert step.status.text == "Model loaded from consumo energía.onnx. The description has unsaved changes."
        assert window.feedback.text == step.status.text

        await window.click(step.save_button)
        assert chooser.await_args.kwargs == {
            "save": True,
            "title": "Save model",
            "extensions": (".onnx",),
            "suggested_name": "consumo energía.onnx",
            "initial_directory": str(model_file.parent),
        }
        errors.assert_not_called()
        assert notices.call_args.kwargs == {"title": SAVED_TITLE}
        assert not step.has_unsaved_changes and step.status.text == "Model saved to consumo energía.onnx"
        assert step.formula.text == FORMULA and _showing_loaded_model(window), "The saved model still has no data."

    asyncio.run(exercise())
    after = _metadata(model_file)
    assert after["description"] == "Predice la energía de cada vivienda."
    assert {key: value for key, value in after.items() if key != "description"} == {
        key: value for key, value in before.items() if key != "description"
    }


# ----------------------------------------------------------------------------
# Replacing the model, and replacing other work with it
# ----------------------------------------------------------------------------


def test_loading_a_dataset_shows_the_data_sections_again_after_confirming_unsaved_changes(
    window: ModelWindow, model_file: Path, dataset_file: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """A loaded model with an edited description is only discarded once the user confirms it."""
    _, _, confirm = dialogs
    chooser = _choose(monkeypatch, model_file, dataset_file, dataset_file)

    async def exercise() -> None:
        await _load_from_start(window)
        loaded = window.model
        await window.type_description("Cambiada")

        confirm.return_value = False
        await _open_dataset_step(window)
        assert "description of the current model" in confirm.call_args.args[0]
        assert chooser.await_count == 2 and window.dataset is None
        assert _showing_loaded_model(window) and window.model is loaded
        assert window.model_step.description.text == "Cambiada"

        confirm.return_value = True
        await window.click(window.dataset_step.open_dataset)
        assert window.dataset is not None and window.model is None
        assert window.wizard.visible and window.preview.visible and not window.no_data_panel.visible
        assert window.wizard.selected_step is window.dataset_step and window.wizard.linear
        assert not window.model_step.has_unsaved_changes and window.predict_step.model is None

    asyncio.run(exercise())


def test_replacing_a_model_without_unsaved_changes_asks_nothing(
    window: ModelWindow, model_file: Path, dataset_file: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """A loaded model that was not edited, or was saved after editing it, is replaced directly."""
    _, _, confirm = dialogs
    _choose(monkeypatch, model_file, model_file, dataset_file)

    async def exercise() -> None:
        await _load_from_start(window)
        await window.type_description("Nueva descripción")
        await window.click(window.model_step.save_button)
        await _open_dataset_step(window)
        assert window.dataset is not None and window.model is None

    asyncio.run(exercise())
    confirm.assert_not_awaited()


def test_loading_a_model_over_an_unsaved_created_model_asks_first(
    window: ModelWindow, model_file: Path, dataset_file: Path, monkeypatch: pytest.MonkeyPatch, dialogs: tuple[Mock, Mock, AsyncMock]
) -> None:
    """An unsaved created model and its dataset are only discarded after confirming."""
    _, _, confirm = dialogs
    _choose(monkeypatch, dataset_file, model_file, model_file)

    async def exercise() -> None:
        await window.click(window.start_screen.dataset_tile)
        await window.click(window.advance)
        window.columns_step._outputs[2].checked = True
        for index, checkbox in enumerate(window.columns_step._inputs):
            checkbox.checked = index in (0, 1)
        await window.settle()
        for _ in range(3):
            await window.click(window.advance)
        await window.click(window.model_step.create_button)
        created = window.model
        assert created is not None and window.fit_panel.visible and window.load_model_button.visible

        confirm.return_value = False
        await window.click(window.load_model_button)
        assert "has not been saved" in confirm.call_args.args[0]
        assert window.model is created and window.dataset is not None and window.fit_panel.visible

        confirm.return_value = True
        await window.click(window.load_model_button)
        assert _showing_loaded_model(window) and window.model is not created
        assert window.selection is None and window.train_data is None and window.preprocessed_data is None
        assert not window.dataset_grid.rows and window.dataset_step.path_text.text == ""
        assert not window.fit_chart.point_series and not window.fit_panel.visible
        assert window.model_step.description.text == "Predice la energía."

    asyncio.run(exercise())
