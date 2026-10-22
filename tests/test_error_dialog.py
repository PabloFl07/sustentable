"""Tests for reusable error messages, ownership, and dismissal without a native host."""

from collections.abc import Iterator
from unittest.mock import Mock, call, patch

import pytest
from pysual import App, Button, ClickEvent, Container, Label, SubWindow
from pysual.errors import LifecycleError

from sustentable.error_dialog import ErrorDialog


@pytest.fixture
def focus_button() -> Iterator[Mock]:
    """Capture keyboard focus requests without requiring a running window."""
    with patch.object(Button, "focus", autospec=True) as focus:
        yield focus


@pytest.fixture
def show_subwindow(focus_button: Mock) -> Iterator[Mock]:
    """Capture modal presentations while retaining the real error dialog controls."""
    with patch.object(SubWindow, "show", autospec=True) as show:
        yield show


@pytest.mark.parametrize("owner_type", [App, Container])
def test_error_message_is_shown_modally_for_its_parent(
    owner_type: type[App] | type[Container], show_subwindow: Mock, focus_button: Mock
) -> None:
    """The full message is displayed read-only using the attached parent as modal owner."""
    owner = owner_type(width=960, height=640)
    dialog = ErrorDialog(parent=owner, title="Unable to open dataset")
    message = "No se puede leer cañón.csv.\nEl archivo contiene símbolos: ñ, á, € y 日本語."

    assert isinstance(dialog, SubWindow)
    assert dialog.visible is False
    assert dialog.title == "Unable to open dataset"

    def focus_after_presentation(button: Button) -> None:
        """Ensure focus is requested only after modal presentation establishes ownership."""
        assert button is dialog.ok
        show_subwindow.assert_called_once_with(dialog, modal=True, owner=owner)

    focus_button.side_effect = focus_after_presentation
    dialog.show_error(message)

    assert dialog.message_text.text == message
    assert isinstance(dialog.message_text, Label)
    assert dialog.message_text.word_wrap is True
    show_subwindow.assert_called_once_with(dialog, modal=True, owner=owner)
    focus_button.assert_called_once_with(dialog.ok)


def test_ok_closes_the_dialog_and_another_error_reuses_it(show_subwindow: Mock) -> None:
    """Dismissing an error leaves the same control ready to display a new message."""
    owner = App()
    owner.error_dialog = ErrorDialog()
    dialog = owner.error_dialog
    dialog.show_error("The first read failed.")
    dialog.message_area.scroll_y = 100

    assert owner.check_handlers() == ()
    with patch.object(SubWindow, "close", autospec=True) as close:
        dialog.ok_on_click(ClickEvent(source=dialog.ok))
        close.assert_called_once_with(dialog)
    show_subwindow.reset_mock()

    dialog.show_error("The second read failed.\nPlease choose another file.")

    assert owner.error_dialog is dialog
    assert dialog.message_text.text == "The second read failed.\nPlease choose another file."
    assert dialog.message_area.scroll_y == 0
    show_subwindow.assert_called_once_with(dialog, modal=True, owner=owner)


def test_independent_dialogs_keep_their_own_messages_and_owners(show_subwindow: Mock) -> None:
    """Different dialog instances never share message controls or modal ownership."""
    first_owner, second_owner = App(), App()
    first = ErrorDialog(parent=first_owner)
    second = ErrorDialog(parent=second_owner)

    first.show_error("First dataset failed.")
    second.show_error("Second dataset failed.")

    assert first.message_text is not second.message_text
    assert first.ok is not second.ok
    assert first.message_text.text == "First dataset failed."
    assert second.message_text.text == "Second dataset failed."
    assert show_subwindow.call_args_list == [
        call(first, modal=True, owner=first_owner),
        call(second, modal=True, owner=second_owner),
    ]


def test_detached_dialog_requires_an_owner_before_showing(show_subwindow: Mock) -> None:
    """An unattached dialog fails clearly before changing its message or requesting a modal."""
    dialog = ErrorDialog()
    previous_message = dialog.message_text.text

    with pytest.raises(LifecycleError, match="Attach the error dialog to an owner"):
        dialog.show_error("No owner is available.")

    assert dialog.message_text.text == previous_message
    show_subwindow.assert_not_called()
