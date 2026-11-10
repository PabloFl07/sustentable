"""Model creation button, formula and training/test metrics for the model step."""

from typing import ClassVar

from pysual import Button, ClickEvent, Container, Event, Label, ScrollArea, Step, Tokens, UiEvent

from sustentable.core.model.regression import RegressionModel, format_metric

STEP_HEIGHT = 176
_READY = "Ready. The model is fitted with the training subset only and evaluated on both subsets."
_NO_FORMULA = "Create the model to see its formula."
_NO_VALUE = "—"
_NAME_WIDTH = 280
_VALUE_WIDTH = 150
_CELL_SPACING = 8
_TABLE_PADDING = 10


class ModelStep(Step):
    """Request the model, then show its formula and its R² and MSE on the training and test subsets.

    The owner window trains the model when ``create_model`` is emitted and calls ``show_model`` with the result, or
    ``clear_model`` once the data it was trained with changes. ``set_prerequisite`` explains what is still missing to
    create a model; the button is only enabled when nothing is.
    """

    create_model: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, tokens: Tokens, **properties: object) -> None:
        """Place the button and status, the formula, and the metrics table side by side."""
        super().__init__(title="Model", direction="horizontal", height=STEP_HEIGHT, padding=8, spacing=16, **properties)
        self._prerequisite = ""
        self._model: RegressionModel | None = None
        box = tokens.raised

        actions = Container(parent=self, layout="stack", spacing=8, width=220)
        self.create_button = Button(parent=actions, text="Create model", width=200, height=36)
        self.status = Label(parent=actions, word_wrap=True, flex=1)

        formula = Container(parent=self, layout="stack", spacing=4, flex=1)
        self.formula_caption = Label(parent=formula, height=20, font_size=13, foreground=tokens.muted)
        formula_box = ScrollArea(parent=formula, layout="stack", padding=10, background=box, flex=1)
        self.formula = Label(parent=formula_box, word_wrap=True, font_family="mono", background=box)

        table_width = _NAME_WIDTH + 2 * _VALUE_WIDTH + 2 * _CELL_SPACING + 2 * _TABLE_PADDING
        metrics = Container(parent=self, layout="stack", spacing=4, width=table_width)
        Label(parent=metrics, text="METRICS", height=20, font_size=13, foreground=tokens.muted)
        table = Container(parent=metrics, layout="stack", padding=_TABLE_PADDING, spacing=8, background=box, flex=1)
        _row(table, box, ("Metric", "Training", "Test"), foreground=tokens.muted)
        self.r2_values = _row(table, box, ("Coefficient of determination (R²)", _NO_VALUE, _NO_VALUE))
        self.mse_values = _row(table, box, ("Mean squared error (MSE)", _NO_VALUE, _NO_VALUE))
        self._refresh()

    def set_prerequisite(self, problem: str) -> None:
        """Explain what must be done before the model can be created, or pass ``""`` when it can be created."""
        self._prerequisite = problem
        self._refresh()

    def show_model(self, model: RegressionModel) -> None:
        """Show the formula and metrics of a newly created model, replacing the previous one."""
        self._model = model
        self._refresh()

    def clear_model(self) -> None:
        """Stop showing a model whose data has changed."""
        self._model = None
        self._refresh()

    def create_button_on_click(self, event: ClickEvent) -> None:
        """Ask the owning window to create the model."""
        self.create_model.emit(UiEvent(source=self, origin=event.origin))

    def _refresh(self) -> None:
        self.create_button.enabled = not self._prerequisite
        model = self._model
        if model is None:
            self.status.text = self._prerequisite or _READY
            self.formula_caption.text = "FORMULA"
            self.formula.text = _NO_FORMULA
            for cell in (*self.r2_values, *self.mse_values):
                cell.text = _NO_VALUE
            return
        train, test = model.train_metrics, model.test_metrics
        kind = "Simple" if model.is_simple else "Multiple"
        self.status.text = f"Model created successfully. {kind} regression trained on {train.rows:,} rows and tested on {test.rows:,} rows."
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
