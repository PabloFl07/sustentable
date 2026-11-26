"""View of a model loaded from a file, without the sections that work with data."""

from typing import ClassVar

from pysual import Button, ClickEvent, Container, Event, Label, Tokens, UiEvent

from sustentable.core.io.import_model import SavedModel
from sustentable.core.model.regression import RegressionModel
from sustentable.gui.steps.model_step import ModelStep
from sustentable.gui.steps.predict_step import PredictStep


class ModelView(Container):
    """Show a model loaded from a file: its columns, formula, metrics and editable description, and predict with it.

    A model file holds no data, so there is no data table, column selection, split, preprocessing or fit chart. The
    owner window calls ``show_model`` with the loaded model and ``clear_model`` once it is discarded. ``save_model``
    asks the owner to save the model in ``summary`` with its description; ``load_model`` and ``load_dataset`` ask the
    owner to replace it with another model or with a dataset.
    """

    load_model: ClassVar[Event[UiEvent]] = Event(UiEvent)
    load_dataset: ClassVar[Event[UiEvent]] = Event(UiEvent)
    save_model: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, tokens: Tokens, **properties: object) -> None:
        """Place the heading and both replacement buttons above the model summary and the prediction panel."""
        super().__init__(layout="stack", spacing=12, **properties)
        header = Container(parent=self, layout="dock", height=36, spacing=8)
        self.dataset_button = Button(parent=header, text="Load dataset…", dock="right", width=160)
        self.model_button = Button(parent=header, text="Load another model…", dock="right", width=200)
        Label(parent=header, text="Saved model", font_size=20, dock="left")
        hint = "The file holds no data: review the model, edit its description, save it again or make predictions."
        Label(parent=header, text=hint, foreground=tokens.muted, dock="fill")
        self.summary = ModelStep(parent=self, tokens=tokens, creatable=False)
        self.prediction = PredictStep(parent=self, tokens=tokens)

    @property
    def model(self) -> RegressionModel | None:
        """The model shown, or ``None`` while there is none."""
        return self.summary.model

    def show_model(self, saved: SavedModel) -> None:
        """Show a loaded model with its description, ready to predict and without unsaved changes."""
        self.summary.show_loaded(saved.model, saved.description, saved.path)
        self.prediction.show_model(saved.model)

    def clear_model(self) -> None:
        """Stop showing a model that has been replaced."""
        self.summary.clear_model()
        self.prediction.clear_model()

    def dataset_button_on_click(self, event: ClickEvent) -> None:
        """Ask the owning window to replace the model with a dataset."""
        self.load_dataset.emit(UiEvent(source=self, origin=event.origin))

    def model_button_on_click(self, event: ClickEvent) -> None:
        """Ask the owning window to replace the model with another saved model."""
        self.load_model.emit(UiEvent(source=self, origin=event.origin))

    def summary_on_save_model(self, event: UiEvent) -> None:
        """Ask the owning window to save the model shown."""
        self.save_model.emit(UiEvent(source=self, origin=event.origin))
