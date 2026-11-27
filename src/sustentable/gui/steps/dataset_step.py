"""File chooser and column selection for the first wizard step."""

from typing import ClassVar

from pysual import Button, ClickEvent, Container, Event, Label, Step, TextBox, Tokens, UiEvent

from sustentable.gui.steps.columns_step import ColumnsStep


class DatasetStep(Step):
    """Show the selected path and the input/output columns in one step.

    The owner window loads the file when ``browse`` is emitted and confirms the columns.
    ``column_picker`` keeps the X and Y choice consistent and emits ``edited`` when it changes.
    """

    browse: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, tokens: Tokens, **properties: object) -> None:
        """Place the file toolbar above the column picker."""
        super().__init__(title="Dataset", height=340, padding=8, spacing=10, **properties)
        row = Container(parent=self, layout="stack", direction="horizontal", spacing=12, height=36)
        self.path_label = Label(parent=row, text="Dataset path", width=112)
        self.path_text = TextBox(parent=row, placeholder="CSV, Excel or SQLite", read_only=True, flex=1)
        self.open_dataset = Button(parent=row, text="Browse", width=100)
        # A private name keeps the panel unnamed so the window can bind ``columns_step_on_edited``.
        self._columns_panel = ColumnsStep(parent=self, tokens=tokens)

    @property
    def column_picker(self) -> ColumnsStep:
        """Column picker shown under the file path."""
        return self._columns_panel

    def open_dataset_on_click(self, event: ClickEvent) -> None:
        """Request the owning window's file dialog without wiring its child button there."""
        self.browse.emit(UiEvent(source=self, origin=event.origin))
