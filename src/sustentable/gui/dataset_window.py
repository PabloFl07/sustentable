"""Window that loads a dataset, displays it in a PandaGrid, and lets the user pick the regression columns.

Reading stays in ``sustentable.core.readers``, DataFrame display stays in ``sustentable.gui.controls.panda_grid``,
and column selection stays in ``sustentable.gui.column_sidebar``.
``dataset`` on the window is the last frame loaded successfully, and ``selection`` the last confirmed columns.
"""

import asyncio
from collections.abc import Hashable
from contextlib import suppress
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
    Slider,
    TextBox,
    autoconfig,  # noqa: F401  (importing it applies pysual's environment defaults)
    get_theme,
)
from pysual.errors import LifecycleError

from sustentable.core.exceptions import DataImportError
from sustentable.core.readers import CSVFileCheck, ExcelFileCheck, SQLiteFileCheck, get_reader
from sustentable.gui.column_sidebar import ColumnSidebar
from sustentable.gui.controls.data_shuffler_slider import TrainTestSliderController
from sustentable.gui.controls.loading_spinner import LoadingSpinner
from sustentable.gui.controls.panda_grid import PandaGrid
from sustentable.gui.controls.sidebar_button import SidebarButton, with_sidebar_button
from sustentable.gui.error_dialog import ErrorDialog

# Same suffixes the import module accepts, including the ones US-14 lists. .sqlite3 is here because SQLiteFileCheck reads it.
DATASET_EXTENSIONS: tuple[str, ...] = (
    *CSVFileCheck.extensions,
    *ExcelFileCheck.extensions,
    *SQLiteFileCheck.extensions,
)


class DatasetWindow(App):
    """Dataset browser with a column side panel.

    ``dataset`` is the last frame loaded, or ``None`` when nothing has loaded. ``selection`` is the last confirmed
    (input columns, output column) pair for that frame, or ``None`` until the user confirms one.
    """

    dataset: DataFrame | None
    selection: tuple[list[Hashable], Hashable] | None

    def build(self) -> None:
        """Create the path toolbar, the column side panel next to the scrolling table, and the reusable error dialog."""
        self.title = "Sustentable"
        self.width = 1280
        self.height = 720
        self.layout = "stack"
        self.padding = 16
        self.spacing = 12
        self.theme = with_sidebar_button(self.theme)
        self.dataset = None
        self.selection = None
        self._loading = False
        self.path_row = Container(layout="stack", direction="horizontal", spacing=12, height=36)
        self.path_label = Label(parent=self.path_row, text="Dataset path")
        self.path_text = TextBox(parent=self.path_row, placeholder="No file selected", read_only=True, flex=1)
        self.open_dataset = Button(parent=self.path_row, text="Browse", width=100)
        self.loading_row = Container(layout="stack", direction="horizontal", spacing=8, height=24, visible=False)
        self.loading_spinner = LoadingSpinner(parent=self.loading_row, width=24, height=24)
        self.loading_label = Label(parent=self.loading_row, text="Loading dataset…", flex=1)
        self.body = Container(layout="stack", direction="horizontal", flex=1, spacing=12)
        self.column_sidebar = ColumnSidebar(parent=self.body, tokens=self.theme.tokens, visible=False)
        self.content = Container(parent=self.body, layout="stack", flex=1, spacing=8)

        self.slider_row = Container(parent=self.content, layout="stack", direction="horizontal", spacing=12, height=36, visible=False)
        self.slider_label = Label(parent=self.slider_row, text="Train: 80.0% | Test: 20.0%", width=200)
        self.split_slider = Slider(parent=self.slider_row, minimum=0, maximum=100, value=20.0, flex=1)
        self.split_slider.changed.connect(self.on_slider_changed)
        self.confirm_split_btn = Button(parent=self.slider_row, text="Confirm", width=100)
        self.confirm_split_btn.click.connect(self.on_confirm_split)

        self.show_sidebar = SidebarButton(parent=self.content, icon="menu", width=28, height=28, tooltip="Show panel", visible=False)
        self.dataset_grid = PandaGrid(parent=self.content, flex=1, min_height=240)
        self.error_dialog = ErrorDialog(title="Unable to open dataset")
        self.column_sidebar.hide_panel.click.connect(self.hide_column_sidebar)
        self.column_sidebar.confirm.click.connect(self.confirm_selection)

    def hide_column_sidebar(self, event: ClickEvent) -> None:
        """Collapse the column panel so the table takes the whole width."""
        self._set_sidebar_visible(False)

    def show_sidebar_on_click(self, event: ClickEvent) -> None:
        """Bring the column panel back."""
        self._set_sidebar_visible(True)

    def _set_sidebar_visible(self, visible: bool) -> None:
        """Show either the column panel or the button that reopens it, never both."""
        self.column_sidebar.visible = visible
        self.show_sidebar.visible = not visible

    def confirm_selection(self, event: ClickEvent) -> None:
        """Keep the chosen columns in ``selection`` for the regression step."""
        inputs, output = self.column_sidebar.selection
        if not inputs or output is None:
            return
        self.selection = (inputs, output)
        self.slider_row.visible = True

    def on_slider_changed(self, event: ChangeEvent[float]) -> None:
        """Updates the data proportion when the user uses the slidebar."""
        if hasattr(self, "split_controller"):
            current_value = self.split_slider.value
            train_pct, test_pct = self.split_controller.on_slider_move(current_value)
            self.slider_label.text = f"Train: {train_pct:.1f}% | Test: {test_pct:.1f}%"
            if hasattr(self, "confirm_split_btn"):
                self.confirm_split_btn.text = "Confirm"
                self.confirm_split_btn.enabled = True

    def on_confirm_split(self, event: ClickEvent) -> None:
        """Confirms the partition and saves the final subsets."""
        if hasattr(self, "split_controller"):
            train_df, test_df = self.split_controller.get_datasets()
            self.train_data = train_df
            self.test_data = test_df
            self.confirm_split_btn.text = "Confirmed!"
            self.confirm_split_btn.enabled = False

    async def open_dataset_on_click(self, event: ClickEvent) -> None:
        """Ask for a dataset file. Cancelling leaves the path, the table, and ``dataset`` as they were."""
        try:
            chosen = await self.choose_path(title="Open dataset", extensions=DATASET_EXTENSIONS)
        except (CapabilityError, OSError, ValueError) as error:
            # The chooser can reject a selected extension before the import module receives the path.
            self.error_dialog.show_error(str(error))
            return
        except LifecycleError:
            return
        if chosen is None:
            return
        await self.show_dataset(chosen)

    async def show_dataset(self, path: str | Path) -> None:
        """Load ``path`` without blocking the UI, keeping the previous dataset if loading fails.

        Ignore overlapping requests while a load is in progress. File reading and grid conversion run in worker
        threads; only completed results are published on the UI thread. Closing the window cancels the wait.
        """
        if self._loading:
            return
        self._loading = True
        animation: asyncio.Task[None] | None = None
        try:
            self.open_dataset.enabled = False
            self.loading_row.visible = True
            animation = asyncio.create_task(self.loading_spinner.animate())
            absolute_path, frame = await asyncio.to_thread(_read_dataset, path)
            await self.dataset_grid.load_dataframe_async(frame)
            self.column_sidebar.load_columns(frame.columns)
            self.dataset = frame
            self.selection = None

            self.split_controller = TrainTestSliderController(frame=self.dataset, initial_test_pct=self.split_slider.value)

            self.path_text.text = absolute_path
            self._set_sidebar_visible(True)
        except (DataImportError, OSError, TypeError, ValueError) as error:
            self.error_dialog.show_error(str(error))
        finally:
            try:
                if animation is not None:
                    animation.cancel()
                    with suppress(asyncio.CancelledError):
                        await animation
            finally:
                self._loading = False
                try:
                    self.loading_row.visible = False
                    self.open_dataset.enabled = True
                except LifecycleError:
                    # Shutdown may already have disposed of the controls while cancelling the pending load.
                    pass


def _read_dataset(path: str | Path) -> tuple[str, DataFrame]:
    """Resolve and read a dataset in a worker without accessing any UI controls."""
    return str(Path(path).resolve()), get_reader(path).read()


def main() -> None:
    """Open the dataset window and wait until it closes."""
    ui = DatasetWindow()
    ui.theme = get_theme("macos")
    ui.run_blocking()
