"""File chooser for the first wizard step."""

from typing import ClassVar

from pysual import Button, ClickEvent, Container, Event, Label, Step, TextBox, UiEvent


class DatasetStep(Step):
    """Show the selected dataset path.

    The owner window loads the file when ``browse`` is emitted. Column selection is shown beside the preview,
    not in this step, so the lists can use the height of the table.
    """

    browse: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, **properties: object) -> None:
        """Place the file toolbar on one row."""
        super().__init__(title="Dataset", padding=8, spacing=8, **properties)
        row = Container(parent=self, layout="stack", direction="horizontal", spacing=12, height=36)
        self.path_label = Label(parent=row, text="Dataset path", width=112)
        self.path_text = TextBox(parent=row, placeholder="CSV, Excel or SQLite", read_only=True, flex=1)
        self.open_dataset = Button(parent=row, text="Browse", width=100)

    def open_dataset_on_click(self, event: ClickEvent) -> None:
        """Request the owning window's file dialog without wiring its child button there."""
        self.browse.emit(UiEvent(source=self, origin=event.origin))
