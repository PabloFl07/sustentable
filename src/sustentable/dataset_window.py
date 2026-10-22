"""Window that loads a dataset and shows every row in a scrolling table.

Reading stays in ``sustentable.data_import``. This module only asks for a path, calls that reader, and displays the result.
``dataset`` on the window is the last frame loaded successfully, ready for the following steps.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Literal

from pandas import DataFrame, Series, isna
from pandas.api.types import is_any_real_numeric_dtype, is_integer_dtype
from pysual import (
    App,
    Button,
    CapabilityError,
    ClickEvent,
    DataGrid,
    GridColumn,
    GridRow,
    TextBox,
    autoconfig,  # noqa: F401  (importing it applies pysual's environment defaults)
    get_theme,
)
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
    """Return columns and rows for ``frame``, using its column names as headers.

    A column of integers or real floats is ``kind="number"``, which is what makes the grid align it to the right.
    Booleans, dates, text, and mixed columns stay text. Missing values become empty cells.
    """
    # iloc picks one column even when two headers share a name.
    kinds = tuple(_column_kind(frame.iloc[:, index]) for index in range(frame.shape[1]))
    columns = tuple(
        GridColumn(f"c{index}", _header(name), kind=kind) for index, (name, kind) in enumerate(zip(frame.columns, kinds, strict=True))
    )
    rows = tuple(
        GridRow(str(index), tuple(_cell(kind, value) for kind, value in zip(kinds, row, strict=True)))
        for index, row in enumerate(frame.itertuples(index=False, name=None))
    )
    return columns, rows


def _column_kind(column: Series) -> Literal["text", "number"]:
    """Return ``number`` when every present value is an integer or a real float, and ``text`` otherwise."""
    if _is_number_dtype(column.dtype):
        return "number"
    found = False
    for value in column:
        if _missing(value):
            continue
        if not _is_number_dtype(type(value)):
            return "text"
        found = True
    # A column with nothing in it has no numbers to align.
    return "number" if found else "text"


def _is_number_dtype(dtype: object) -> bool:
    """Return whether ``dtype`` is an integer or real-float dtype, including numpy scalars.

    A number cell only accepts ``int`` and ``float``, so booleans and complex values stay out.
    """
    return is_any_real_numeric_dtype(dtype)


def _header(name: object) -> str:
    """Return a column title on one line. A blank name stays blank."""
    return " ".join(str(name).split())


def _cell(kind: Literal["text", "number"], value: object) -> str | int | float | None:
    """Return one grid cell. Missing values are ``None``, and numbers are Python ``int`` or ``float``."""
    if _missing(value):
        return None
    if kind == "number":
        return _number(value)
    return _text(value)


def _missing(value: object) -> bool:
    """Return whether ``value`` is missing or is a non-finite float."""
    if value is None or (isinstance(value, float) and not isfinite(value)):
        return True
    try:
        missing = isna(value)
    except TypeError, ValueError:
        return False
    return missing is True


def _number(value: object) -> int | float | None:
    """Return ``value`` as a Python ``int`` or ``float``. A non-finite float is missing."""
    if not _is_number_dtype(type(value)):
        raise TypeError(f"Cannot show {value!r} as a number.")
    if is_integer_dtype(type(value)):
        return int(value)
    number = float(value)
    if not isfinite(number):
        return None
    return number


def _text(value: object) -> str:
    """Return the text of one cell, rejecting text the grid cannot store."""
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
