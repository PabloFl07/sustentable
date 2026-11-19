"""Tests for reusable confirmation questions without a native host."""

import asyncio
from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest
from pysual import App, Button, ClickEvent, SubWindow
from pysual.errors import LifecycleError
from pysual.presentation import Outcome

from sustentable.gui.controls.confirm_dialog import ConfirmDialog


@pytest.fixture
def focus_button() -> Iterator[Mock]:
    """Capture keyboard focus requests without requiring a running window."""
    with patch.object(Button, "focus", autospec=True) as focus:
        yield focus


def _presentation(result: object) -> Mock:
    presentation = Mock()

    async def wait_async() -> Outcome:
        return Outcome(result)

    presentation.wait_async = wait_async
    return presentation


@pytest.mark.parametrize(("result", "expected"), [(True, True), (False, False), (None, False)])
def test_ask_returns_true_only_when_confirmed(focus_button: Mock, result: object, expected: bool) -> None:
    """Confirming returns True; cancelling or closing the window with its close button returns False."""
    owner = App(width=960, height=640)
    owner.dialog = ConfirmDialog()
    dialog = owner.dialog
    with patch.object(SubWindow, "show", autospec=True, return_value=_presentation(result)) as show:
        answer = asyncio.run(dialog.ask("¿Reemplazar «modelo.onnx»?", title="Replace", confirm_text="Replace"))
    assert answer is expected
    show.assert_called_once_with(dialog, modal=True, owner=owner)
    focus_button.assert_called_once_with(dialog.cancel)
    assert dialog.title == "Replace" and dialog.confirm.text == "Replace" and dialog.cancel.text == "Cancel"
    assert dialog.message_text.text == "¿Reemplazar «modelo.onnx»?"


def test_buttons_close_with_their_answer() -> None:
    """Each button closes the dialog with the answer it stands for."""
    dialog = ConfirmDialog()
    with patch.object(SubWindow, "close", autospec=True) as close:
        dialog.confirm_on_click(ClickEvent(source=dialog.confirm))
        dialog.cancel_on_click(ClickEvent(source=dialog.cancel))
    assert [call.args for call in close.call_args_list] == [(dialog, True), (dialog, False)]


def test_ask_requires_an_owner() -> None:
    """A detached dialog cannot be shown modally."""
    with pytest.raises(LifecycleError):
        asyncio.run(ConfirmDialog().ask("Continue?"))
