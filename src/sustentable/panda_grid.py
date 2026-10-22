"""A Pysual data grid that displays pandas DataFrames as typed, read-only cells."""

from __future__ import annotations

import asyncio
from itertools import repeat
from math import isfinite
from typing import Literal

from pandas import DataFrame, Series, isna
from pandas.api.types import is_any_real_numeric_dtype, is_float, is_integer_dtype, is_scalar
from pysual import DataGrid, GridColumn, GridRow


class PandaGrid(DataGrid):
    """Display a DataFrame using DataGrid's scrolling, sorting, and numeric alignment.

    Pass a frame at construction or call ``load_dataframe`` to replace it later. The grid stores a display snapshot;
    it never modifies the source frame. Keep the original frame in the application for subsequent data processing.
    Mutating the frame does not refresh the grid: call ``load_dataframe`` again to display those changes.
    Ordinary DataGrid properties, such as ``flex`` and ``min_height``, are accepted as keyword arguments.
    """

    def __init__(self, dataframe: DataFrame | None = None, **properties: object) -> None:
        """Create a grid, optionally populated from ``dataframe`` before it is attached to its parent."""
        if dataframe is not None:
            if "columns" in properties or "rows" in properties:
                raise TypeError("Pass either a dataframe or raw columns and rows, not both.")
            columns, rows = _grid_records(dataframe)
            properties.update(columns=columns, rows=rows)
        super().__init__(**properties)

    def load_dataframe(self, dataframe: DataFrame) -> None:
        """Replace every column and row, retaining the previous grid if conversion or validation fails.

        Missing values and non-finite floats become blank cells. Real numeric columns retain numeric sorting;
        other values become text. Pysual's data-size and cell-length limits apply, without truncating data.
        """
        columns, rows = _grid_records(dataframe)
        self.set_data(columns=columns, rows=rows)

    async def load_dataframe_async(self, dataframe: DataFrame) -> None:
        """Prepare a display snapshot in a worker and publish it on the UI thread.

        Use this from async event handlers to keep the UI responsive during conversion. Conversion or validation
        failures leave the previous grid unchanged; cancelling the wait discards the worker's eventual result.
        """
        columns, rows = await asyncio.to_thread(_grid_records, dataframe)
        self.set_data(columns=columns, rows=rows)


def _grid_records(frame: DataFrame) -> tuple[tuple[GridColumn, ...], tuple[GridRow, ...]]:
    """Convert the entire frame before publishing any changes to the control."""
    if not isinstance(frame, DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    # Use positions for keys and dtype lookup: neither column labels nor the index need to be unique.
    kinds = tuple(_column_kind(frame.iloc[:, index]) for index in range(frame.shape[1]))
    columns = tuple(
        GridColumn(f"c{index}", " ".join(str(name).split()), kind=kind)
        for index, (name, kind) in enumerate(zip(frame.columns, kinds, strict=True))
    )
    values = frame.itertuples(index=False, name=None) if kinds else repeat((), len(frame))
    rows = tuple(
        GridRow(str(index), tuple(_cell(kind, value) for kind, value in zip(kinds, row, strict=True))) for index, row in enumerate(values)
    )
    return columns, rows


def _column_kind(column: Series) -> Literal["text", "number"]:
    """Recognize real numeric dtypes and object columns containing only real numbers and missing values."""
    if is_any_real_numeric_dtype(column.dtype):
        return "number"
    found = False
    for value in column:
        if _missing(value):
            continue
        if not is_any_real_numeric_dtype(type(value)):
            return "text"
        found = True
    return "number" if found else "text"


def _missing(value: object) -> bool:
    """Recognize missing scalars and non-finite floats without treating array-valued cells as booleans."""
    if not is_scalar(value):
        return False
    return bool(isna(value)) or (is_float(value) and not isfinite(value))


def _cell(kind: Literal["text", "number"], value: object) -> str | int | float | None:
    """Normalize pandas and NumPy scalars to the Python values accepted by GridRow."""
    if _missing(value):
        return None
    if kind == "number":
        if is_integer_dtype(type(value)):
            return int(value)
        number = float(value)
        return number if isfinite(number) else None
    return str(value)
