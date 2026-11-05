
"""Database statistics and configuration controls for the dataset step."""

from typing import ClassVar
from sustentable.core.data_processing import detect_nan_values, process_nan_values, mean, median,constant,delete_strategy

from pysual import (
    Container,
    Dropdown,
    Event,
    Label,
    Step,
    TextBox,
    UiEvent,
)


class DatabaseStep(Step):
    """Display database column statistics and collect configuration options."""

    option_changed: ClassVar[Event[UiEvent]] = Event(UiEvent)
    custom_value_changed: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, **properties: object) -> None:
        """Build the database statistics and configuration controls."""
        super().__init__(title="Database analysis", padding=8, **properties)

        # Database statistics
        statistics = Container(
            parent=self,
            layout="stack",
            direction="horizontal",
            spacing=12,
            height=36,
        )

        self.total_columns_label = Label(
            parent=statistics,
            text="Total columns: 0",
            flex=1,
        )

        self.empty_columns_label = Label(
            parent=statistics,
            text="Empty columns: 0",
            flex=1,
        )

        # Configuration dropdown
        options = Container(
            parent=self,
            layout="stack",
            direction="horizontal",
            spacing=12,
            height=36,
        )

        self.option_label = Label(
            parent=options,
            text="Select option:",
            width=112,
        )

        self.option_dropdown = Dropdown(
            parent=options,
            items=("mean", "median", "constant", "delete"),
            width=180,
        )

        # Conditional custom value input
        self.custom_container = Container(
            parent=self,
            layout="stack",
            direction="horizontal",
            spacing=12,
            height=36,
        )

        self.custom_label = Label(
            parent=self.custom_container,
            text="Custom value:",
            width=112,
        )

        self.custom_text = TextBox(
            parent=self.custom_container,
            placeholder="Enter a value...",
            flex=1,
        )

        # Initially hidden
        self.custom_container.visible = False

    def update_statistics(
        self,
        total_columns: int,
        empty_columns: int,
    ) -> None:
        """Update the database statistics displayed by the step."""
        self.total_columns_label.text = (
            f"Total columns: {total_columns}"
        )

        self.empty_columns_label.text = (
            f"Empty columns: {empty_columns}"
        )

    def option_dropdown_on_changed(self, event: UiEvent) -> None:
        """Notify the owning window when the selected option changes."""
        self.option_changed.emit(
            UiEvent(source=self, origin=event.origin)
        )

    def custom_text_on_changed(self, event: UiEvent) -> None:
        """Notify the owning window when the custom value changes."""
        self.custom_value_changed.emit(
            UiEvent(source=self, origin=event.origin)
        )

    def show_custom_input(self, visible: bool) -> None:
        """Show or hide the custom value input."""
        self.custom_container.visible = visible
