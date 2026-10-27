"""Pysual proof of concept for the Sustentable GUI.

Run it with ``uv run python prototipos/pysual_poc.py``. The application in ``src/sustentable`` does not import this module.
"""

from __future__ import annotations

from pysual import (
    App,
    Button,
    CapabilityError,
    ClickEvent,
    DataGrid,
    Dropdown,
    GridColumn,
    GridRow,
    Label,
    TextBox,
    get_theme,
)

_COLUMNS: tuple[GridColumn, ...] = (
    GridColumn("area", "Area", width=120, kind="number"),
    GridColumn("rooms", "Rooms", width=120, kind="number"),
    GridColumn("price", "Price", width=140, kind="number"),
)

_SAMPLE_ROWS: tuple[GridRow, ...] = (
    GridRow("row-1", (85, 3, 210_000)),
    GridRow("row-2", (62, 2, 165_000)),
    GridRow("row-3", (110, 4, 280_000)),
    GridRow("row-4", (48, 1, 132_000)),
    GridRow("row-5", (96, 3, 241_000)),
    GridRow("row-6", (74, 2, 188_000)),
    GridRow("row-7", (130, 5, 325_000)),
    GridRow("row-8", (55, 2, 149_000)),
)


class GuiPrototype(App):
    """Window that tries the controls the project needs from Pysual."""

    def build(self) -> None:
        """Create the file button, a scrolling sample table, a dropdown, a text field, and a result."""
        self.title = "Sustentable GUI prototype"
        self.width = 720
        self.height = 640
        self.layout = "stack"
        self.padding = 16
        self.spacing = 8
        self.path_label = Label(text="No file selected")
        self.open_file = Button(text="Select file")
        self.sample_grid = DataGrid(columns=_COLUMNS, rows=_SAMPLE_ROWS, height=160)
        self.model_kind = Dropdown(
            items=("Simple regression", "Multiple regression", "Prediction"),
            selected_index=0,
        )
        self.entry = TextBox(placeholder="Enter a value")
        self.show_value = Button(text="Show value")
        self.result_label = Label(text="The chosen values appear here.")

    async def open_file_on_click(self, event: ClickEvent) -> None:
        """Show the chosen file path. Cancelling leaves the current text in place."""
        try:
            chosen = await self.open_text_file()
        except (CapabilityError, OSError, UnicodeError, ValueError) as error:
            self.path_label.text = str(error)
            return
        if chosen is None:
            return
        self.path_label.text = chosen.location or chosen.name

    def show_value_on_click(self, event: ClickEvent) -> None:
        """Show the dropdown selection and the text entered in the field."""
        self.result_label.text = f"Selected: {self.model_kind.selected_item}. Entered: {self.entry.text}"


def main() -> None:
    """Open the prototype and wait until it closes."""
    ui = GuiPrototype()
    ui.theme = get_theme("macos")
    ui.run_blocking()


if __name__ == "__main__":
    main()
