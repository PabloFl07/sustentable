"""Compact file chooser controls for the dataset step."""

from typing import ClassVar

from pysual import Button, ClickEvent, Container, Event, Label, Step, TextBox, UiEvent


class DatasetStep(Step):
    """Keep the selected path visible while the window owns loading and errors."""

    browse: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, **properties: object) -> None:
        """Build the file toolbar using the application's existing reader formats."""
        super().__init__(title="Dataset", padding=8, **properties)
        row = Container(parent=self, layout="stack", direction="horizontal", spacing=12, height=36)
        self.path_label = Label(parent=row, text="Dataset path", width=112)
        self.path_text = TextBox(parent=row, placeholder="CSV, Excel or SQLite", read_only=True, flex=1)
        self.open_dataset = Button(parent=row, text="Browse", width=100)

    def open_dataset_on_click(self, event: ClickEvent) -> None:
        """Request the owning window's file dialog without wiring its child button there."""
        self.browse.emit(UiEvent(source=self, origin=event.origin))
