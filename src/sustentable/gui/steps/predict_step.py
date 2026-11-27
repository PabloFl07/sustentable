"""Input fields, prediction button and result for the prediction step."""

from pysual import Button, ClickEvent, Container, Label, ScrollArea, Step, TextBox, Tokens

from sustentable.core.model.prediction import PredictionInputError, Predictor, format_prediction, predict_value

STEP_HEIGHT = 200
_NO_MODEL = "Create or load a model to make predictions."
_READY = "Enter a value for each input. Decimals can use a point or a comma, such as 3.5 or 3,5."
_NO_RESULT = "—"
_FIELD_WIDTH = 200
_ROW_HEIGHT = 36
_LABEL_WIDTH = 140
_FORM_WIDTH = _LABEL_WIDTH + _FIELD_WIDTH + 44
_RESULT_WIDTH = 320
_CAPTION_HEIGHT = 20


class PredictStep(Step):
    """Show one field per input of the current model and predict its output for the values typed in them.

    The step only needs a model, so it works the same with a model created from a dataset and with a model loaded
    from a file: the owner window calls ``show_model`` whenever a model becomes available and ``clear_model`` once it
    is discarded. Each call regenerates the fields and clears the previous result. The prediction itself is made by
    ``predict_value``, which does not depend on the GUI.
    """

    def __init__(self, tokens: Tokens, **properties: object) -> None:
        """Place the inputs and the prediction button in one form, with the result beside it."""
        super().__init__(title="Predict", optional=True, direction="horizontal", height=STEP_HEIGHT, padding=8, spacing=16, **properties)
        self._model: Predictor | None = None
        self._fields: list[TextBox] = []
        self._rows: list[Container] = []
        self._error = False
        self._colors = tokens.foreground, tokens.danger
        box, muted = tokens.raised, tokens.muted

        form = Container(parent=self, layout="stack", spacing=8, width=_FORM_WIDTH)
        self.inputs_caption = Label(parent=form, text="INPUTS", height=_CAPTION_HEIGHT, font_size=13, foreground=muted)
        self.field_list = ScrollArea(parent=form, layout="stack", padding=8, spacing=6, background=box, flex=1)
        self.predict_button = Button(parent=form, text="Make prediction", height=36)

        result_panel = Container(parent=self, layout="stack", spacing=4, width=_RESULT_WIDTH)
        self.result_caption = Label(parent=result_panel, text="PREDICTION", height=_CAPTION_HEIGHT, font_size=13, foreground=muted)
        result_box = Container(parent=result_panel, layout="stack", padding=10, background=box, height=44)
        self.result = Label(parent=result_box, text=_NO_RESULT, font_family="mono", background=box)
        self.status = Label(parent=result_panel, word_wrap=True, flex=1)
        Container(parent=self, flex=1)
        self._refresh()

    @property
    def model(self) -> Predictor | None:
        """The model predictions are made with, or ``None`` while there is none."""
        return self._model

    @property
    def fields(self) -> dict[str, TextBox]:
        """The input field of each input column, keyed by the text of its label, in the order of the model's inputs."""
        if self._model is None:
            return {}
        return {str(column): field for column, field in zip(self._model.inputs, self._fields, strict=True)}

    @property
    def has_error(self) -> bool:
        """Whether the status reports a problem with the values typed in the fields."""
        return self._error

    def show_model(self, model: Predictor) -> None:
        """Generate one empty field per input of ``model`` and clear the previous prediction."""
        self._model = model
        self._build_fields()
        self._refresh()

    def clear_model(self) -> None:
        """Remove the fields and the prediction of a model that is no longer available."""
        self._model = None
        self._build_fields()
        self._refresh()

    def predict_button_on_click(self, event: ClickEvent) -> None:
        """Predict the output for the typed values, or explain which values are missing or not numbers."""
        model = self._model
        if model is None:
            return
        values = {column: field.text for column, field in zip(model.inputs, self._fields, strict=True)}
        try:
            value = predict_value(model, values)
        except PredictionInputError as error:
            self.result.text = _NO_RESULT
            self._show_status(str(error), error=True)
            return
        except ValueError as error:
            self.result.text = _NO_RESULT
            self._show_status(f"The prediction could not be made: {error}", error=True)
            return
        self.result.text = format_prediction(model.output, value)
        self._show_status(f"Predicted {model.output} for the values entered.", error=False)

    def _build_fields(self) -> None:
        """Replace the input rows with one labeled field per input of the current model."""
        for row in self._rows:
            row.destroy()
        self._rows, self._fields = [], []
        if self._model is None:
            return
        for column in self._model.inputs:
            row = Container(parent=self.field_list, layout="stack", direction="horizontal", spacing=12, height=_ROW_HEIGHT)
            Label(parent=row, text=str(column), width=_LABEL_WIDTH)
            field = TextBox(parent=row, placeholder="e.g. 3.5", width=_FIELD_WIDTH)
            self._rows.append(row)
            self._fields.append(field)

    def _show_status(self, text: str, error: bool) -> None:
        """Show ``text`` below the result, in the danger color when it reports a problem."""
        self._error = error
        self.status.text = text
        self.status.foreground = self._colors[error]

    def _refresh(self) -> None:
        model = self._model
        self.predict_button.enabled = model is not None
        self.result.text = _NO_RESULT
        if model is None:
            self.inputs_caption.text = "INPUTS"
            self.result_caption.text = "PREDICTION"
            self._show_status(_NO_MODEL, error=False)
            return
        self.inputs_caption.text = f"INPUTS · {len(model.inputs)}"
        self.result_caption.text = f"PREDICTION · {model.output}"
        self._show_status(_READY, error=False)
