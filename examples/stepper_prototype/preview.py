"""The same tiny sample grid for shared and in-step previews."""

from pysual import Container, DataGrid, GridColumn, GridRow

SAMPLE_ROWS = tuple(
    GridRow(str(i), row)
    for i, row in enumerate(
        ((85, 2, 18.5, 120), (112, 3, 19.2, 165), (68, 1, 17.8, 95), (140, 4, 21.1, 210), (95, 2, 20.3, 145), (120, 3, 18.9, 180))
    )
)


def sample_grid(parent: Container, **properties: object) -> DataGrid:
    """Build a read-only preview with consistent columns and sizing."""
    return DataGrid(
        parent=parent,
        columns=tuple(GridColumn(name, name, kind="number") for name in ("Area", "Occupants", "Temperature", "Energy")),
        row_height=28,
        header_height=32,
        **properties,
    )
