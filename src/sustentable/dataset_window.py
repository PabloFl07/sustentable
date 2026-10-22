"""Window that loads a dataset and displays it in a PandaGrid.

Reading stays in ``sustentable.data_import`` and DataFrame display stays in ``sustentable.panda_grid``.
``dataset`` on the window is the last frame loaded successfully, ready for the following steps.
"""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame
from pysual import (
    App,
    Button,
    CapabilityError,
    ClickEvent,
    TextBox,
    autoconfig,  # noqa: F401  (importing it applies pysual's environment defaults)
    get_theme,
)
from pysual.errors import LifecycleError

from sustentable.data_import import CSVFileCheck, ExcelFileCheck, SQLiteFileCheck, get_reader
from sustentable.exceptions import DataImportError
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
        """Create the open button, the path, the error text, and an empty scrolling table."""
        self.title = "Sustentable"
        self.width = 960
        self.height = 640
        self.layout = "stack"
        self.padding = 16
        self.spacing = 8
        self.dataset = None
        self.open_dataset = Button(text="Open dataset")
        self.path_text = TextBox(text="No file selected", read_only=True)
        self.status_text = TextBox(text="", read_only=True, multiline=True, height=56)
        self.dataset_grid = PandaGrid(flex=1, min_height=240)

    async def open_dataset_on_click(self, event: ClickEvent) -> None:
        """Ask for a dataset file. Cancelling leaves the path, the table, and ``dataset`` as they were."""
        try:
            chosen = await self.choose_path(title="Open dataset", extensions=DATASET_EXTENSIONS)
        except CapabilityError as error:
            self.status_text.text = str(error)
            return
        except LifecycleError:
            return
        if chosen is None:
            return
        self.show_dataset(chosen)

    def show_dataset(self, path: str | Path) -> None:
        """Load ``path`` into the table. A failed load keeps the previous dataset and shows the error message."""
        try:
            absolute_path = str(Path(path).resolve())
            frame = get_reader(path).read()
            self.dataset_grid.load_dataframe(frame)
        except (DataImportError, OSError, TypeError, ValueError) as error:
            self.status_text.text = str(error)
            return
        self.dataset = frame
        self.path_text.text = absolute_path
        self.status_text.text = ""


def main() -> None:
    """Open the dataset window and wait until it closes."""
    ui = DatasetWindow()
    ui.theme = get_theme("macos")
    ui.run_blocking()
