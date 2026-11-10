"""Guided dataset workflow with modular forms and one persistent data preview."""

import asyncio
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from pandas import DataFrame, concat
from pysual import (
    App,
    Button,
    CapabilityError,
    ChangeEvent,
    ClickEvent,
    Container,
    Label,
    Step,
    Stepper,
    UiEvent,
    autoconfig,  # noqa: F401 (importing it applies pysual's environment defaults)
    get_theme,
)
from pysual.errors import LifecycleError

from sustentable.core.io.exceptions import DataImportError
from sustentable.core.io.readers import SUPPORTED_EXTENSIONS, get_reader
from sustentable.core.model.regression import FitPlot, RegressionModel, create_model, fit_plot
from sustentable.core.processing.data_processing import process_nan_values
from sustentable.gui.controls.error_dialog import ErrorDialog
from sustentable.gui.controls.loading_spinner import LoadingSpinner
from sustentable.gui.controls.panda_grid import PandaGrid
from sustentable.gui.controls.scatter_chart import LineSeries, PointSeries, ScatterChart
from sustentable.gui.controls.sidebar_button import SidebarButton, with_sidebar_button
from sustentable.gui.steps.columns_step import ColumnsStep
from sustentable.gui.steps.dataset_step import DatasetStep
from sustentable.gui.steps.model_step import ModelStep
from sustentable.gui.steps.preprocess_step import PreprocessStep
from sustentable.gui.steps.split_step import SplitStep

OPEN_ERROR_TITLE = "Unable to open dataset"
MODEL_ERROR_TITLE = "Unable to create model"
# Fixed colors keep both subsets recognizable in any theme; translucent markers show where points overlap.
TRAIN_COLOR = "#60a5facc"
TEST_COLOR = "#fb923ccc"
LINE_COLOR = "#34d399"


class DatasetWindow(App):
    """Load data, confirm regression columns and retain the confirmed train/test subsets and model.

    ``dataset`` is the last successfully loaded frame. ``selection``, ``train_data`` and ``test_data``
    hold confirmed results, or ``None`` when a prerequisite has changed. ``preprocessed_data`` holds the
    (train, test) subsets without missing values in the selected columns, ready for the model.
    After preprocessing is applied the shared table shows those subsets; it shows the loaded dataset again
    once the result is discarded. ``model`` is the model created from ``preprocessed_data``, kept for saving and
    predictions; while the model step is selected, its fit chart takes the place of the table.
    """

    dataset: DataFrame | None
    selection: tuple[list[Hashable], Hashable] | None
    train_data: DataFrame | None
    test_data: DataFrame | None
    preprocessed_data: tuple[DataFrame, DataFrame] | None
    model: RegressionModel | None

    def build(self) -> None:
        """Place compact horizontal steps above the shared table and a stable navigation footer."""
        self.title, self.width, self.height = "Sustentable", 1280, 720
        self.layout, self.padding, self.spacing = "stack", 16, 12
        self.theme = with_sidebar_button(self.theme)
        self.dataset = self.selection = self.train_data = self.test_data = self.preprocessed_data = self.model = None
        self._busy_message = ""
        self._dataset_name = ""
        self._dataset_caption = "Dataset preview"
        self._showing_processed = False
        self.wizard = Stepper(parent=self, orientation="horizontal", linear=True, header_height=48)
        self.dataset_step = DatasetStep(parent=self.wizard)
        self.columns_step = ColumnsStep(parent=self.wizard, tokens=self.theme.tokens)
        self.split_step = SplitStep(parent=self.wizard)
        self.preprocess_step = PreprocessStep(parent=self.wizard)
        self.model_step = ModelStep(parent=self.wizard, tokens=self.theme.tokens)
        Step(parent=self.wizard, title="Predict", optional=True, height=0, padding=0)
        self.loading_row = Container(parent=self, layout="stack", direction="horizontal", spacing=8, height=24, visible=False)
        self.loading_spinner = LoadingSpinner(parent=self.loading_row, width=24, height=24)
        self.loading_label = Label(parent=self.loading_row, flex=1)
        self.preview = Container(parent=self, layout="stack", flex=1, spacing=8)
        self.preview_label = Label(parent=self.preview, text="Dataset preview", height=24)
        self.dataset_grid = PandaGrid(parent=self.preview, flex=1)
        self.fit_panel = Container(parent=self, layout="stack", flex=1, spacing=8, visible=False)
        self.fit_label = Label(parent=self.fit_panel, height=24)
        self.fit_chart = ScatterChart(parent=self.fit_panel, flex=1)
        footer = Container(parent=self, layout="dock", height=36, spacing=8)
        self.advance = Button(parent=footer, text="Continue", dock="right", width=176)
        self.back = SidebarButton(parent=footer, text="Back", dock="right", width=80)
        self.feedback = Label(parent=self, height=24)
        self.error_dialog = ErrorDialog(title=OPEN_ERROR_TITLE)
        self._refresh()

    async def columns_step_on_edited(self, event: UiEvent) -> None:
        """Invalidate dependent workflow results after the columns step changes."""
        if self._busy_message:
            return
        if self.selection != self.columns_step.selection:
            self._clear_selection()
        self._refresh()
        await self._show_raw_preview()

    async def split_step_on_edited(self, event: UiEvent) -> None:
        """Discard confirmed subsets when their allocation changes."""
        if not self._busy_message:
            self._clear_split()
            self._refresh()
            await self._show_raw_preview()

    async def preprocess_step_on_edited(self, event: UiEvent) -> None:
        """Discard an applied treatment when its options change and missing values must be treated again."""
        if not self._busy_message and self.preprocess_step.has_missing_values:
            self.preprocessed_data = None
            self._clear_model()
            self._refresh()
            await self._show_raw_preview()

    async def preprocess_step_on_apply(self, event: UiEvent) -> None:
        """Treat the missing values of both subsets off the UI thread; keep the previous state if it fails."""
        if self._busy_message or self.train_data is None or self.selection is None:
            return
        train, test, columns = self.train_data, self.test_data, self._selected_columns()
        try:
            strategy, constant = self.preprocess_step.selected_strategy, self.preprocess_step.constant_value
            async with self._busy("Preprocessing data…"):
                result = await asyncio.to_thread(process_nan_values, train, test, columns, strategy, constant)
                preview = await asyncio.to_thread(concat, [result.train, result.test], ignore_index=True)
                self._showing_processed = True
                await self.dataset_grid.load_dataframe_async(preview)
        except (TypeError, ValueError, MemoryError) as error:
            self.preprocessed_data = None
            self._clear_model()
            self.preprocess_step.clear_result()
            self.error_dialog.show_error(str(error) or "Not enough memory to preprocess the dataset.", title="Unable to preprocess data")
            self._refresh()
            await self._show_raw_preview()
            return
        self.preprocessed_data = result.train, result.test
        self._clear_model()
        self.preprocess_step.show_result(result)
        self.preview_label.text = f"{self._dataset_name} · preprocessed · training {len(result.train):,} + test {len(result.test):,} rows"
        self._refresh()

    async def model_step_on_create_model(self, event: UiEvent) -> None:
        """Train the model and prepare its chart off the UI thread; a failure keeps the previous model."""
        if self._busy_message or self.preprocessed_data is None or self.selection is None:
            return
        (train, test), (inputs, output) = self.preprocessed_data, self.selection
        try:
            async with self._busy("Creating model…"):
                model, plot = await asyncio.to_thread(_train_model, train, test, inputs, output)
                self._show_fit(plot)
        except (TypeError, ValueError, MemoryError) as error:
            self.error_dialog.show_error(str(error) or "Not enough memory to create the model.", title=MODEL_ERROR_TITLE)
            return
        self.model = model
        self.model_step.show_model(model)
        kind = "simple" if model.is_simple else "multiple"
        self.fit_label.text = f"{self._dataset_name} · {kind} regression fit · training {len(train):,} + test {len(test):,} rows"
        self._refresh()

    def _show_fit(self, plot: FitPlot) -> None:
        """Draw both subsets and the model line, keeping the previous chart if the data cannot be drawn."""
        self.fit_chart.set_data(
            (PointSeries("Training", plot.train_points, TRAIN_COLOR, "circle"), PointSeries("Test", plot.test_points, TEST_COLOR, "square")),
            (LineSeries(plot.line_label, *plot.line, LINE_COLOR),),
        )
        self.fit_chart.update(x_label=plot.x_label, y_label=plot.y_label)

    async def _show_raw_preview(self) -> None:
        """Reload the loaded dataset into the table when it still shows a discarded preprocessing result."""
        if not self._showing_processed or self.dataset is None:
            return
        self._showing_processed = False
        async with self._busy("Restoring dataset preview…"):
            await self.dataset_grid.load_dataframe_async(self.dataset)
        self.preview_label.text = self._dataset_caption

    def wizard_on_changed(self, event: ChangeEvent[int]) -> None:
        """Keep guidance and navigation in sync with header and keyboard navigation."""
        self._refresh()

    def back_on_click(self, event: ClickEvent) -> None:
        """Return to an earlier step without discarding its choices."""
        self.wizard.previous()

    async def advance_on_click(self, event: ClickEvent) -> None:
        """Confirm the current form before making its dependent steps reachable."""
        if not self.advance.enabled:
            return
        step = self.wizard.selected_step
        if step is self.columns_step and self.selection is None:
            self.selection = self.columns_step.selection
            self.split_step.reset(self.dataset)
        elif step is self.split_step and self.train_data is None:
            try:
                async with self._busy("Splitting dataset…"):
                    self.train_data, self.test_data = await self.split_step.split()
            except (TypeError, ValueError, MemoryError) as error:
                self.error_dialog.show_error(str(error) or "Not enough memory to split the dataset.", title="Unable to split dataset")
                return
            self.preprocess_step.load(self.train_data, self.test_data, self._selected_columns())
            # Without missing values the split is already usable and preprocessing can be skipped.
            if not self.preprocess_step.has_missing_values:
                self.preprocessed_data = self.train_data, self.test_data
        self._refresh()
        self.wizard.next()

    def _refresh(self) -> None:
        self.wizard.enabled = self.dataset_step.open_dataset.enabled = not self._busy_message
        self.back.enabled = not self._busy_message and self.wizard.selected_index > 0
        self.advance.text = "Continue"
        self.feedback.text = "Not implemented yet."
        problem = ""
        for step, issue, confirmed in (
            (self.dataset_step, "Open a dataset to begin." if self.dataset is None else "", self.dataset is not None),
            (self.columns_step, self.columns_step.problem, self.selection is not None),
            (self.split_step, self.split_step.problem, self.train_data is not None),
            (self.preprocess_step, self._preprocess_problem(), self.preprocessed_data is not None),
            (self.model_step, "" if self.model is not None else "Create the model to continue.", self.model is not None),
        ):
            confirmation = f"Confirm {step.title.lower()}"
            hint = issue or ("" if confirmed else f"{confirmation} to continue.")
            # Native header/keyboard navigation uses the same current prerequisites as the footer.
            step.set_validation(hint, show_error=False)
            if step is self.wizard.selected_step:
                problem = issue
                if step is self.split_step:
                    done = "Split confirmed. " + self.split_step.counts.text
                elif step is self.preprocess_step:
                    done = self.preprocess_step.status.text or self.preprocess_step.summary.text
                elif step is self.model_step:
                    done = self.model_step.status.text
                else:
                    done = "Ready to continue."
                self.feedback.text = hint or done
                # The model is confirmed with its own button, so the footer only continues.
                if not confirmed and step is not self.dataset_step and step is not self.model_step:
                    self.advance.text = confirmation
        if self._busy_message:
            self.feedback.text = self._busy_message
        self.advance.enabled = not self._busy_message and not problem and self.wizard.selected_index < len(self.wizard.steps) - 1
        self.model_step.set_prerequisite(self._model_prerequisite())
        showing_fit = self.model is not None and self.wizard.selected_step is self.model_step
        self.preview.visible, self.fit_panel.visible = not showing_fit, showing_fit

    def _preprocess_problem(self) -> str:
        """Block the model while the selected columns still contain missing values."""
        if problem := self.preprocess_step.problem:
            return problem
        if self.preprocessed_data is None:
            return "Apply preprocessing to treat the missing values before creating the model."
        return ""

    def _model_prerequisite(self) -> str:
        """Explain what is still missing before the model can be created."""
        if self.selection is None:
            return "Confirm the input and output columns before creating the model."
        if self.train_data is None:
            return "Confirm the split into training and test subsets before creating the model."
        if self.preprocessed_data is None:
            return "Treat the missing values of the selected columns before creating the model."
        return ""

    def _selected_columns(self) -> list[Hashable]:
        """Confirmed input columns followed by the output column."""
        if self.selection is None:
            return []
        inputs, output = self.selection
        return [*inputs, output]

    def _clear_selection(self) -> None:
        """Discard the confirmed columns and the split that depends on them."""
        self.selection = None
        self._clear_split()
        self.split_step.reset(None)

    def _clear_split(self) -> None:
        """Discard the confirmed subsets and everything derived from them."""
        self.train_data = self.test_data = self.preprocessed_data = None
        self.preprocess_step.load(None, None, [])
        self._clear_model()

    def _clear_model(self) -> None:
        """Discard the model, which no longer matches the data prepared for it."""
        self.model = None
        self.model_step.clear_model()
        self.fit_chart.clear()

    async def dataset_step_on_browse(self, event: UiEvent) -> None:
        """Choose a file, preserving the workflow when the dialog is cancelled or fails."""
        if self._busy_message:
            return
        try:
            chosen = await self.choose_path(title="Open dataset", extensions=SUPPORTED_EXTENSIONS)
        except (CapabilityError, OSError, ValueError) as error:
            self.error_dialog.show_error(str(error), title=OPEN_ERROR_TITLE)
            return
        except LifecycleError:
            return
        if chosen is not None:
            await self.show_dataset(chosen)

    async def show_dataset(self, path: str | Path) -> None:
        """Read and convert off the UI thread; retain the previous workflow if either stage fails.

        Overlapping requests are ignored. Cancelling the wait discards the worker's eventual result.
        """
        if self._busy_message:
            return
        async with self._busy("Loading dataset…"):
            try:
                absolute_path, frame = await asyncio.to_thread(_read_dataset, path)
                await self.dataset_grid.load_dataframe_async(frame)
                self.columns_step.load_columns(frame.columns)
                self.dataset = frame
                self._clear_selection()
                self.dataset_step.path_text.text = absolute_path
                self._showing_processed = False
                self._dataset_name = Path(absolute_path).name
                self._dataset_caption = f"{self._dataset_name} · {len(frame):,} rows · {len(frame.columns):,} columns"
                self.preview_label.text = self._dataset_caption
                self.wizard.reset()
            except (DataImportError, OSError, TypeError, ValueError) as error:
                self.error_dialog.show_error(str(error), title=OPEN_ERROR_TITLE)

    @asynccontextmanager
    async def _busy(self, message: str) -> AsyncIterator[None]:
        """Share activity feedback and cancellation cleanup for imports and confirmed splits."""
        self._busy_message = message
        self.loading_label.text = message
        self.loading_row.visible = True
        self._refresh()
        animation = asyncio.create_task(self.loading_spinner.animate())
        try:
            yield
        finally:
            try:
                animation.cancel()
                with suppress(asyncio.CancelledError):
                    await animation
            finally:
                self._busy_message = ""
                try:
                    self.loading_row.visible = False
                    self._refresh()
                except LifecycleError:
                    # Teardown may already have disposed the controls while cancelling a pending load.
                    pass


def _read_dataset(path: str | Path) -> tuple[str, DataFrame]:
    """Resolve and read a dataset in a worker without accessing UI controls."""
    return str(Path(path).resolve()), get_reader(path).read()


def _train_model(train: DataFrame, test: DataFrame, inputs: list[Hashable], output: Hashable) -> tuple[RegressionModel, FitPlot]:
    """Create the model and its chart data in a worker without accessing UI controls."""
    model = create_model(train, test, inputs, output)
    return model, fit_plot(model, train, test)


def main() -> None:
    """Open the dataset window and wait until it closes."""
    ui = DatasetWindow()
    ui.theme = get_theme("modern_dark")
    ui.run_blocking()
