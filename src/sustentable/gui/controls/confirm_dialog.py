"""Reusable, owner-modal yes/no questions for Pysual applications."""

from pysual import Button, ClickEvent, Container, Label, ScrollArea, SubWindow
from pysual.errors import LifecycleError


class ConfirmDialog(SubWindow):
    """Ask the user to confirm an action, retaining the dialog for reuse.

    Attach the dialog to a window or container, then ``await ask(message)``. It returns ``True`` only when the
    confirmation button is pressed; cancelling or closing the dialog returns ``False``.
    """

    def __init__(self, **properties: object) -> None:
        """Create the message area and both buttons inside an initially hidden subwindow."""
        defaults: dict[str, object] = {
            "title": "Confirm",
            "layout": "stack",
            "width": 560,
            "height": 220,
            "padding": 16,
            "spacing": 12,
            "visible": False,
        }
        defaults.update(properties)
        super().__init__(**defaults)
        self.message_area = ScrollArea(layout="stack", padding=0, flex=1)
        self.message_text = Label(parent=self.message_area, word_wrap=True)
        self.actions = Container(layout="dock", height=36, spacing=8)
        self.confirm = Button(parent=self.actions, text="OK", dock="right", width=160)
        self.cancel = Button(parent=self.actions, text="Cancel", dock="right", width=100)

    async def ask(self, message: str, title: str | None = None, confirm_text: str = "OK", cancel_text: str = "Cancel") -> bool:
        """Show ``message`` modally and wait, without blocking the UI event loop, until the user answers."""
        owner = self.parent
        if owner is None:
            raise LifecycleError("Attach the confirmation dialog to an owner before showing it.")
        if title is not None:
            self.title = title
        self.message_text.text = message
        self.confirm.text, self.cancel.text = confirm_text, cancel_text
        self.message_area.update(scroll_x=0, scroll_y=0)
        area = owner.content_bounds(owner.bounds)
        width = self.width if self.width is not None else self.bounds.width
        height = self.height if self.height is not None else self.bounds.height
        self.update(left=max(0, (area.width - width) / 2), top=max(0, (area.height - height) / 2))
        presentation = self.show(modal=True, owner=owner)
        # The safe choice gets the focus, so pressing Enter by accident never confirms.
        self.cancel.focus()
        outcome = await presentation.wait_async()
        return outcome.result is True

    def confirm_on_click(self, event: ClickEvent) -> None:
        """Accept the action."""
        self.close(True)

    def cancel_on_click(self, event: ClickEvent) -> None:
        """Reject the action."""
        self.close(False)
