"""Reusable, owner-modal error messages for Pysual applications."""

from pysual import Button, ClickEvent, Container, Label, ScrollArea, SubWindow
from pysual.errors import LifecycleError


class ErrorDialog(SubWindow):
    """Display an error message with an OK button, retaining the dialog for reuse.

    Attach the dialog to a window or container, then call ``show_error(message)``.
    The parent supplies the modal owner and centering area. Normal SubWindow properties can override the defaults.
    """

    def __init__(self, **properties: object) -> None:
        """Create the message area and dismissal button inside an initially hidden subwindow."""
        defaults: dict[str, object] = {
            "title": "Error",
            "layout": "stack",
            "width": 560,
            "height": 240,
            "padding": 16,
            "spacing": 12,
            "visible": False,
        }
        defaults.update(properties)
        super().__init__(**defaults)
        self.message_area = ScrollArea(layout="stack", padding=0, flex=1)
        self.message_text = Label(parent=self.message_area, word_wrap=True)
        self.actions = Container(layout="dock", height=36)
        self.ok = Button(parent=self.actions, text="OK", dock="right", width=100)

    def show_error(self, message: str) -> None:
        """Replace the message and open modally without blocking the UI event loop."""
        owner = self.parent
        if owner is None:
            raise LifecycleError("Attach the error dialog to an owner before showing it.")
        self.message_text.text = message
        self.message_area.update(scroll_x=0, scroll_y=0)
        area = owner.content_bounds(owner.bounds)
        width = self.width if self.width is not None else self.bounds.width
        height = self.height if self.height is not None else self.bounds.height
        self.update(left=max(0, (area.width - width) / 2), top=max(0, (area.height - height) / 2))
        self.show(modal=True, owner=owner)
        self.ok.focus()

    def ok_on_click(self, event: ClickEvent) -> None:
        """Dismiss the current error and return interaction to the owner."""
        self.close()
