"""Model creation button, formula, saving and training/test metrics for the model step."""

from pathlib import Path
from typing import ClassVar

from pysual import Button, ChangeEvent, ClickEvent, Container, Event, Label, ScrollArea, Step, TextBox, Tokens, UiEvent

from sustentable.core.model.regression import RegressionModel, format_metric

STEP_HEIGHT = 176
_READY = "Ready. The model is fitted with the training subset only and evaluated on both subsets."
_NO_FORMULA = "Create the model to see its formula."
_NO_VALUE = "—"
_ACTIONS_WIDTH = 230
_SAVE_WIDTH = 260
_NAME_WIDTH = 200
_VALUE_WIDTH = 112
_CELL_SPACING = 8
_TABLE_PADDING = 10
_CAPTION_HEIGHT = 20


class ModelStep(Step):
    """Request the model, then show its formula, a description to save with it, and its R² and MSE on both subsets.

    The owner window trains the model when ``create_model`` is emitted and calls ``show_model`` with the result, or
    ``clear_model`` once the data it was trained with changes. ``set_prerequisite`` explains what is still missing to
    create a model; the create button is only enabled when nothing is. ``save_model`` asks the owner to save the model
    shown with ``description_text``; its button is only enabled while a model is shown, and ``show_saved`` reports where
    it was saved.

    ``show_loaded`` shows a model loaded from a file instead. There is no data behind it, so the create button is
    replaced by the input and output columns, which the columns step cannot show. Clearing a loaded model also clears
    the description that came with it, so it is never saved with a model created afterwards.

    ``has_unsaved_changes`` tells whether the model shown was never saved or its description changed since it was
    saved or loaded, and ``saved_path`` is that file. ``edited`` is emitted whenever the description changes.
    """

    create_model: ClassVar[Event[UiEvent]] = Event(UiEvent)
    save_model: ClassVar[Event[UiEvent]] = Event(UiEvent)
    edited: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, tokens: Tokens, **properties: object) -> None:
        """Place the create button or the columns, the status, the formula, the metrics table and the save panel."""
        super().__init__(title="Model", direction="horizontal", height=STEP_HEIGHT, padding=8, spacing=16, **properties)
        self._prerequisite = ""
        self._model: RegressionModel | None = None
        self._loaded = False
        self._saved_path: Path | None = None
        # The description written to ``_saved_path``, or ``None`` while the model shown has not been saved.
        self._saved_description: str | None = None
        # Whether ``_saved_path`` was written by saving the model, rather than read when it was loaded.
        self._written = False
        box, muted = tokens.raised, tokens.muted

        actions = Container(parent=self, layout="stack", spacing=4, width=_ACTIONS_WIDTH)
        Label(parent=actions, text="MODEL", height=_CAPTION_HEIGHT, font_size=13, foreground=muted)
        actions_box = Container(parent=actions, layout="stack", padding=10, spacing=8, background=box, flex=1)
        self.create_button = Button(parent=actions_box, text="Create model", height=36)
        self.columns_text = Label(parent=actions_box, word_wrap=True, background=box)
        self.status = Label(parent=actions_box, word_wrap=True, background=box, flex=1)

        formula = Container(parent=self, layout="stack", spacing=4, flex=1)
        self.formula_caption = Label(parent=formula, height=_CAPTION_HEIGHT, font_size=13, foreground=muted)
        formula_box = ScrollArea(parent=formula, layout="stack", padding=10, background=box, flex=1)
        self.formula = Label(parent=formula_box, word_wrap=True, font_family="mono", background=box)

        table_width = _NAME_WIDTH + 2 * _VALUE_WIDTH + 2 * _CELL_SPACING + 2 * _TABLE_PADDING
        metrics = Container(parent=self, layout="stack", spacing=4, width=table_width)
        Label(parent=metrics, text="METRICS", height=_CAPTION_HEIGHT, font_size=13, foreground=muted)
        table = Container(parent=metrics, layout="stack", padding=_TABLE_PADDING, spacing=8, background=box, flex=1)
        _row(table, box, ("Metric", "Training", "Test"), foreground=muted)
        self.r2_values = _row(table, box, ("Determination (R²)", _NO_VALUE, _NO_VALUE))
        self.mse_values = _row(table, box, ("Mean squared error", _NO_VALUE, _NO_VALUE))

        save = Container(parent=self, layout="stack", spacing=4, width=_SAVE_WIDTH)
        Label(parent=save, text="DESCRIPTION", height=_CAPTION_HEIGHT, font_size=13, foreground=muted)
        self.description = TextBox(
            parent=save, multiline=True, word_wrap=True, placeholder="Optional: what the model predicts and how to use it.", flex=1
        )
        self.save_button = Button(parent=save, text="Save model", width=_SAVE_WIDTH, height=32)
        self._refresh()

    @property
    def model(self) -> RegressionModel | None:
        """The model shown, or ``None`` while there is none."""
        return self._model

    @property
    def loaded(self) -> bool:
        """Whether the model shown was loaded from a file, so there is no data behind it."""
        return self._loaded

    @property
    def description_text(self) -> str:
        """The description written by the user, without surrounding blank space. It may be empty."""
        return self.description.text.strip()

    @property
    def saved_path(self) -> Path | None:
        """The file the model shown was last saved to or loaded from, or ``None`` while it has not been saved."""
        return self._saved_path

    @property
    def has_unsaved_changes(self) -> bool:
        """Whether the model shown was never saved, or its description changed since it was last saved or loaded."""
        return self._model is not None and self.description_text != self._saved_description

    def set_prerequisite(self, problem: str) -> None:
        """Explain what must be done before the model can be created, or pass ``""`` when it can be created."""
        self._prerequisite = problem
        self._refresh()

    def show_model(self, model: RegressionModel) -> None:
        """Show the formula and metrics of a newly created model, replacing the previous one; it starts unsaved."""
        self._show(model, loaded=False, saved_path=None, saved_description=None)

    def show_loaded(self, model: RegressionModel, description: str, path: Path) -> None:
        """Show a model loaded from ``path`` with the ``description`` saved with it, which starts without unsaved changes."""
        self.description.text = description
        self._show(model, loaded=True, saved_path=path, saved_description=self.description_text)

    def clear_model(self) -> None:
        """Stop showing a model whose data has changed or that has been replaced."""
        self._show(None, loaded=False, saved_path=None, saved_description=None)

    def show_saved(self, path: Path, description: str) -> None:
        """Report that the model shown was saved to ``path`` with ``description``; editing it again is an unsaved change."""
        self._saved_path, self._saved_description, self._written = path, description, True
        self._refresh()

    def create_button_on_click(self, event: ClickEvent) -> None:
        """Ask the owning window to create the model."""
        self.create_model.emit(UiEvent(source=self, origin=event.origin))

    def save_button_on_click(self, event: ClickEvent) -> None:
        """Ask the owning window to save the model shown."""
        if self._model is not None:
            self.save_model.emit(UiEvent(source=self, origin=event.origin))

    def description_on_changed(self, event: ChangeEvent[str]) -> None:
        """Report in the status whether the edited description still has to be saved."""
        self._refresh()
        self.edited.emit(UiEvent(source=self, origin=event.origin))

    def _show(self, model: RegressionModel | None, loaded: bool, saved_path: Path | None, saved_description: str | None) -> None:
        if self._loaded and not loaded:
            self.description.text = ""
        self._model, self._loaded, self._saved_path, self._saved_description = model, loaded, saved_path, saved_description
        self._written = False
        self._refresh()

    def _refresh(self) -> None:
        self.create_button.visible = not self._loaded
        self.columns_text.visible = self._loaded
        self.create_button.enabled = not self._prerequisite
        model = self._model
        self.save_button.enabled = model is not None
        if model is None:
            self.status.text = self._prerequisite or _READY
            self.status.tooltip = ""
            self.columns_text.text = ""
            self.formula_caption.text = "FORMULA"
            self.formula.text = _NO_FORMULA
            for cell in (*self.r2_values, *self.mse_values):
                cell.text = _NO_VALUE
            return
        train, test = model.train_metrics, model.test_metrics
        kind = "Simple" if model.is_simple else "Multiple"
        if self._saved_path is None:
            self.status.text = f"Model created successfully. {kind} regression trained on {train.rows:,} rows and tested on {test.rows:,} rows."
        else:
            # Next to the columns there is only room for the file name; the tooltip keeps the full path.
            place = self._saved_path.name if self._loaded else self._saved_path
            self.status.text = f"Model {'saved to' if self._written else 'loaded from'} {place}"
            if self.has_unsaved_changes:
                self.status.text += ". The description has unsaved changes."
        self.status.tooltip = "" if self._saved_path is None else str(self._saved_path)
        self.columns_text.text = f"Output (Y): {model.output}\nInputs (X): {', '.join(str(column) for column in model.inputs)}"
        self.formula_caption.text = f"FORMULA · {kind.upper()} REGRESSION"
        self.formula.text = model.formula
        self.r2_values[0].text, self.r2_values[1].text = format_metric(train.r2), format_metric(test.r2)
        self.mse_values[0].text, self.mse_values[1].text = format_metric(train.mse), format_metric(test.mse)


def _row(table: Container, background: str, texts: tuple[str, str, str], foreground: str | None = None) -> tuple[Label, Label]:
    """Add a metric name and its training and test cells to ``table``; return both value cells."""
    row = Container(parent=table, layout="stack", direction="horizontal", spacing=_CELL_SPACING, height=22)
    widths = (_NAME_WIDTH, _VALUE_WIDTH, _VALUE_WIDTH)
    _, train, test = (
        Label(parent=row, text=text, width=width, background=background, foreground=foreground)
        for text, width in zip(texts, widths, strict=True)
    )
    return train, test
