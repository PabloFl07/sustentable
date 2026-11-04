"""Guided dataset workflow with modular forms and one persistent data preview."""

import asyncio
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from pandas import DataFrame
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

from sustentable.core.exceptions import DataImportError
from sustentable.core.readers import CSVFileCheck, ExcelFileCheck, SQLiteFileCheck, get_reader
from sustentable.gui.columns_step import ColumnsStep
from sustentable.gui.controls.loading_spinner import LoadingSpinner
from sustentable.gui.controls.panda_grid import PandaGrid
from sustentable.gui.controls.sidebar_button import SidebarButton, with_sidebar_button
from sustentable.gui.dataset_step import DatasetStep
from sustentable.gui.error_dialog import ErrorDialog
from sustentable.gui.split_step import SplitStep

DATASET_EXTENSIONS: tuple[str, ...] = (
    *CSVFileCheck.extensions,
    *ExcelFileCheck.extensions,
    *SQLiteFileCheck.extensions,
)


class DatasetWindow(App):
    """Load data, confirm regression columns and retain the confirmed train/test subsets.

    ``dataset`` is the last successfully loaded frame. ``selection``, ``train_data`` and ``test_data``
    hold confirmed results, or ``None`` when a prerequisite has changed.
    """

    dataset: DataFrame | None
    selection: tuple[list[Hashable], Hashable] | None
    train_data: DataFrame | None
    test_data: DataFrame | None

    def build(self) -> None:
        """Place compact horizontal steps above the shared table and a stable navigation footer."""
        self.title, self.width, self.height = "Sustentable", 1280, 720
        self.layout, self.padding, self.spacing = "stack", 16, 12
        self.theme = with_sidebar_button(self.theme)
        self.dataset = self.selection = self.train_data = self.test_data = None
        self._busy_message = ""
        self.wizard = Stepper(parent=self, orientation="horizontal", linear=True, header_height=48)
        self.dataset_step = DatasetStep(parent=self.wizard)
        self.columns_step = ColumnsStep(parent=self.wizard, tokens=self.theme.tokens)
        self.split_step = SplitStep(parent=self.wizard)
        for title in ("Preprocess", "Model", "Predict"):
            Step(parent=self.wizard, title=title, optional=True, height=0, padding=0)
        self.loading_row = Container(parent=self, layout="stack", direction="horizontal", spacing=8, height=24, visible=False)
        self.loading_spinner = LoadingSpinner(parent=self.loading_row, width=24, height=24)
        self.loading_label = Label(parent=self.loading_row, flex=1)
        preview = Container(parent=self, layout="stack", flex=1, spacing=8)
        self.preview_label = Label(parent=preview, text="Dataset preview", height=24)
        self.dataset_grid = PandaGrid(parent=preview, flex=1)
        footer = Container(parent=self, layout="dock", height=36, spacing=8)
        self.advance = Button(parent=footer, text="Continue", dock="right", width=176)
        self.back = SidebarButton(parent=footer, text="Back", dock="right", width=80)
        self.feedback = Label(parent=self, height=24)
        self.error_dialog = ErrorDialog(title="Unable to open dataset")
        self._refresh()

    def columns_step_on_edited(self, event: UiEvent) -> None:
        """Invalidate dependent workflow results after the columns step changes."""
        if self._busy_message:
            return
        if self.selection != self.columns_step.selection:
            self.selection = self.train_data = self.test_data = None
            self.split_step.reset(None)
        self._refresh()

    def split_step_on_edited(self, event: UiEvent) -> None:
        """Discard confirmed subsets when their allocation changes."""
        if not self._busy_message:
            self.train_data = self.test_data = None
            self._refresh()

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
                self.error_dialog.title = "Unable to split dataset"
                self.error_dialog.show_error(str(error) or "Not enough memory to split the dataset.")
                return
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
        ):
            confirmation = f"Confirm {step.title.lower()}"
            hint = issue or ("" if confirmed else f"{confirmation} to continue.")
            # Native header/keyboard navigation uses the same current prerequisites as the footer.
            step.set_validation(hint, show_error=False)
            if step is self.wizard.selected_step:
                problem = issue
                self.feedback.text = hint or (
                    "Split confirmed. " + self.split_step.counts.text if step is self.split_step else "Ready to continue."
                )
                if not confirmed and step is not self.dataset_step:
                    self.advance.text = confirmation
        if self._busy_message:
            self.feedback.text = self._busy_message
        self.advance.enabled = not self._busy_message and not problem and self.wizard.selected_index < len(self.wizard.steps) - 1

    async def dataset_step_on_browse(self, event: UiEvent) -> None:
        """Choose a file, preserving the workflow when the dialog is cancelled or fails."""
        if self._busy_message:
            return
        self.error_dialog.title = "Unable to open dataset"
        try:
            chosen = await self.choose_path(title="Open dataset", extensions=DATASET_EXTENSIONS)
        except (CapabilityError, OSError, ValueError) as error:
            self.error_dialog.show_error(str(error))
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
        self.error_dialog.title = "Unable to open dataset"
        async with self._busy("Loading dataset…"):
            try:
                absolute_path, frame = await asyncio.to_thread(_read_dataset, path)
                await self.dataset_grid.load_dataframe_async(frame)
                self.columns_step.load_columns(frame.columns)
                self.dataset = frame
                self.selection = self.train_data = self.test_data = None
                self.split_step.reset(None)
                self.dataset_step.path_text.text = absolute_path
                self.preview_label.text = f"{Path(absolute_path).name} · {len(frame):,} rows · {len(frame.columns):,} columns"
                self.wizard.reset()
            except (DataImportError, OSError, TypeError, ValueError) as error:
                self.error_dialog.show_error(str(error))

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


def main() -> None:
    """Open the dataset window and wait until it closes."""
    ui = DatasetWindow()
    ui.theme = get_theme("modern_dark")
    ui.run_blocking()
