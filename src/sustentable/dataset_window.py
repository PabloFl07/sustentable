"""Window that loads a dataset and shows every row in a scrolling table.

Reading stays in ``sustentable.data_import``. This module only asks for a path, calls that reader, and displays the result.
``dataset`` on the window is the last frame loaded successfully, ready for the following steps.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from pandas import DataFrame, isna
from pysual import autoconfig, get_theme, App, Button, CapabilityError, ClickEvent, DataGrid, GridColumn, GridRow, TextBox
from pysual.errors import LifecycleError

from sustentable.data_import import CSVFileCheck, ExcelFileCheck, SQLiteFileCheck, get_reader
from sustentable.exceptions import DataImportError

# Same suffixes the import module accepts, including the ones US-14 lists. .sqlite3 is here because SQLiteFileCheck reads it.
DATASET_EXTENSIONS: tuple[str, ...] = (
    *CSVFileCheck.extensions,
    *ExcelFileCheck.extensions,
    *SQLiteFileCheck.extensions,
)

# DataGrid rejects a text cell longer than this. The message stays in the window instead of closing it.
_MAX_CELL_LENGTH = 4096


def grid_records(frame: DataFrame) -> tuple[tuple[GridColumn, ...], tuple[GridRow, ...]]:
    """Return text columns and rows for ``frame``, using its column names as headers.

    Missing values become empty cells. Every other value is shown with ``str``, so numpy numbers and dates fit the grid.
    """
    columns = tuple(GridColumn(f"c{index}", _header(name)) for index, name in enumerate(frame.columns))
    rows = tuple(
        GridRow(str(index), tuple(_cell_text(value) for value in row)) for index, row in enumerate(frame.itertuples(index=False, name=None))
    )
    return columns, rows


def _header(name: object) -> str:
    """Return a column title on one line. A blank name stays blank."""
    return " ".join(str(name).split())


def _cell_text(value: object) -> str | None:
    """Return the text of one cell, or ``None`` when the value is missing."""
    if value is None or (isinstance(value, float) and not isfinite(value)):
        return None
    try:
        missing = isna(value)
    except TypeError, ValueError:
        missing = False
    if missing is True:
        return None
    text = str(value)
    if len(text) > _MAX_CELL_LENGTH:
        raise ValueError(f"A value is longer than {_MAX_CELL_LENGTH} characters and cannot be shown.")
    return text


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
        self.dataset_grid = DataGrid(flex=1, min_height=240)

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
        """Load ``path`` into the table. A failed load keeps the previous dataset and shows the reader's message."""
        try:
            frame = get_reader(path).read()
            columns, rows = grid_records(frame)
            self.dataset_grid.set_data(columns=columns, rows=rows)
        except DataImportError as error:
            self.status_text.text = str(error)
            return
        except (TypeError, ValueError) as error:
            self.status_text.text = str(error)
            return
        self.dataset = frame
        self.path_text.text = str(Path(path).resolve())
        self.status_text.text = ""


def main() -> None:
    """Open the dataset window and wait until it closes."""
    ui = DatasetWindow()
    ui.theme = get_theme("macos")
    ui.run_blocking()
