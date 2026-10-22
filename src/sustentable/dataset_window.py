"""Window that loads a dataset and displays it in a PandaGrid.

Reading stays in ``sustentable.data_import`` and DataFrame display stays in ``sustentable.panda_grid``.
``dataset`` on the window is the last frame loaded successfully, ready for the following steps.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

from pandas import DataFrame
from pysual import (
    App,
    Button,
    CapabilityError,
    ClickEvent,
    Container,
    Label,
    TextBox,
    autoconfig,  # noqa: F401  (importing it applies pysual's environment defaults)
    get_theme,
)
from pysual.errors import LifecycleError

from sustentable.data_import import CSVFileCheck, ExcelFileCheck, SQLiteFileCheck, get_reader
from sustentable.error_dialog import ErrorDialog
from sustentable.exceptions import DataImportError
from sustentable.loading_spinner import LoadingSpinner
from sustentable.panda_grid import PandaGrid

# Same suffixes the import module accepts, including the ones US-14 lists. .sqlite3 is here because SQLiteFileCheck reads it.
DATASET_EXTENSIONS: tuple[str, ...] = (
    *CSVFileCheck.extensions,
    *ExcelFileCheck.extensions,
    *SQLiteFileCheck.extensions,
)


class DatasetWindow(App):
    """Dataset browser. ``dataset`` is the last frame loaded, or ``None`` when nothing has loaded."""

    dataset: DataFrame | None

    def build(self) -> None:
        """Create the path toolbar, scrolling table, and reusable modal error dialog."""
        self.title = "Sustentable"
        self.width = 960
        self.height = 640
        self.layout = "stack"
        self.padding = 16
        self.spacing = 12
        self.dataset = None
        self._loading = False
        self.path_row = Container(layout="stack", direction="horizontal", spacing=12, height=36)
        self.path_label = Label(parent=self.path_row, text="Dataset path")
        self.path_text = TextBox(parent=self.path_row, placeholder="No file selected", read_only=True, flex=1)
        self.open_dataset = Button(parent=self.path_row, text="Browse", width=100)
        self.loading_row = Container(layout="stack", direction="horizontal", spacing=8, height=24, visible=False)
        self.loading_spinner = LoadingSpinner(parent=self.loading_row, width=24, height=24)
        self.loading_label = Label(parent=self.loading_row, text="Loading dataset…", flex=1)
        self.dataset_grid = PandaGrid(flex=1, min_height=240)
        self.error_dialog = ErrorDialog(title="Unable to open dataset")

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
            self.dataset = frame
            self.path_text.text = absolute_path
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
