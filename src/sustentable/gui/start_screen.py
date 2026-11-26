"""Start screen that offers to begin from a saved model or from a dataset."""

from typing import ClassVar

from pysual import ClickEvent, Container, Event, Label, Tokens, UiEvent

from sustentable.gui.controls.action_tile import ActionTile

TILE_WIDTH = 340
TILE_HEIGHT = 260


class StartScreen(Container):
    """Show two large choices: recover a previously saved model, or load a dataset to create a new model.

    The owner window handles ``load_model`` and ``load_dataset``, and replaces this screen with the matching
    sections once a file loads.
    """

    load_model: ClassVar[Event[UiEvent]] = Event(UiEvent)
    load_dataset: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, tokens: Tokens, **properties: object) -> None:
        """Center the heading and both choices in the available space."""
        super().__init__(layout="stack", spacing=12, **properties)
        Container(parent=self, flex=1)
        Label(parent=_centered(self, 44), text="How do you want to start?", font_size=30)
        subtitle = "Open a model you saved before, or load a dataset to create a new one."
        Label(parent=_centered(self, 24), text=subtitle, foreground=tokens.muted)
        Container(parent=self, height=16)
        tiles = _centered(self, TILE_HEIGHT, spacing=24)
        self.model_tile = ActionTile(
            parent=tiles,
            text="Load model",
            description="Review a previously saved model, edit its description and make predictions.",
            icon="layers",
            width=TILE_WIDTH,
        )
        self.dataset_tile = ActionTile(
            parent=tiles,
            text="Load dataset",
            description="Start from a CSV, Excel or SQLite file to create a new model.",
            icon="grid",
            width=TILE_WIDTH,
        )
        Container(parent=self, flex=1)

    def model_tile_on_click(self, event: ClickEvent) -> None:
        """Ask the owning window to load a saved model."""
        self.load_model.emit(UiEvent(source=self, origin=event.origin))

    def dataset_tile_on_click(self, event: ClickEvent) -> None:
        """Ask the owning window to load a dataset."""
        self.load_dataset.emit(UiEvent(source=self, origin=event.origin))


def _centered(parent: Container, height: float, spacing: float = 0) -> Container:
    """Add a row of ``height`` to ``parent`` and return the box in its middle, as wide as the controls added to it."""
    row = Container(parent=parent, layout="stack", direction="horizontal", height=height)
    Container(parent=row, flex=1)
    box = Container(parent=row, layout="stack", direction="horizontal", spacing=spacing)
    Container(parent=row, flex=1)
    return box
