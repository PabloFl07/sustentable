"""Input and output column lists, shown beside the dataset preview."""

from collections import Counter
from collections.abc import Hashable, Iterable
from itertools import count
from typing import ClassVar

from pysual import ChangeEvent, CheckBox, Container, Event, Label, RadioButton, ScrollArea, Tokens, UiEvent

HEADER_HEIGHT = 20
ROW_HEIGHT = 28
ROW_SPACING = 4
SUMMARY_HEIGHT = 168


class ColumnsStep(Container):
    """List the dataset columns as X check boxes and Y radio buttons, and summarize the chosen columns.

    The owner window confirms the selection; this panel keeps the choices consistent and emits ``edited``.
    A column chosen as Y cannot also be an X. Column labels are kept as they are in the DataFrame,
    so the selection can index the frame directly.

    Controls are kept by column position, so repeated labels each get their own row. When two labels print
    the same, the shown text adds the column's 1-based position so the user can tell them apart.
    """

    _instances = count()
    edited: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, tokens: Tokens, **properties: object) -> None:
        """Stack two narrow lists and keep the selection summary underneath them."""
        super().__init__(layout="stack", spacing=8, flex=1, **properties)
        self._background = tokens.raised
        self._tokens = tokens
        # Each panel gets its own radio group so two panels never share the Y choice.
        self._output_group_name = f"column-step-output-{next(self._instances)}"
        self._columns: list[Hashable] = []
        self._texts: list[str] = []
        self._inputs: list[CheckBox] = []
        self._outputs: list[RadioButton] = []

        lists = Container(parent=self, layout="stack", direction="horizontal", spacing=8, flex=1)
        inputs = Container(parent=lists, layout="stack", spacing=6, flex=1)
        Label(parent=inputs, text="INPUTS (X)", height=HEADER_HEIGHT, font_size=13, foreground=tokens.muted)
        self.input_group = ScrollArea(parent=inputs, layout="stack", padding=8, spacing=ROW_SPACING, background=tokens.raised, flex=1)
        outputs = Container(parent=lists, layout="stack", spacing=6, flex=1)
        Label(parent=outputs, text="OUTPUT (Y)", height=HEADER_HEIGHT, font_size=13, foreground=tokens.muted)
        self.output_group = ScrollArea(parent=outputs, layout="stack", padding=8, spacing=ROW_SPACING, background=tokens.raised, flex=1)

        self.mode = Label(parent=self, text="", height=HEADER_HEIGHT, font_size=14)
        box = tokens.raised
        self.summary_box = ScrollArea(parent=self, layout="stack", padding=10, spacing=6, background=box, max_height=SUMMARY_HEIGHT)
        Label(parent=self.summary_box, text="OUTPUT (Y)", height=20, font_size=13, foreground=tokens.muted, background=box)
        self.output_chips = Container(parent=self.summary_box, layout="flow", direction="horizontal", spacing=6)
        self.input_caption = Label(parent=self.summary_box, text="INPUTS (X)", height=20, font_size=13, foreground=tokens.muted, background=box)
        self.input_chips = Container(parent=self.summary_box, layout="flow", direction="horizontal", spacing=6)

        self._refresh_mode()

    def load_columns(self, columns: Iterable[Hashable]) -> None:
        """Replace the listed columns, clearing the previous selection."""
        for control in [*self._inputs, *self._outputs]:
            control.destroy()
        self._columns = list(columns)
        self._texts = _distinct_texts(self._columns)
        self._inputs = []
        self._outputs = []

        for text in self._texts:
            # These controls are created from dataset columns, without fixed names for convention binding.
            check = CheckBox(parent=self.input_group, text=text, height=ROW_HEIGHT, background=self._background, font_size=14)
            check.changed.connect(self._on_selection)
            self._inputs.append(check)

            radio = RadioButton(
                parent=self.output_group, text=text, group=self._output_group_name, height=ROW_HEIGHT, background=self._background, font_size=14
            )
            radio.changed.connect(self._on_selection)
            self._outputs.append(radio)

        self._refresh_mode()

    @property
    def input_columns(self) -> list[Hashable]:
        """Columns checked as inputs, in dataset order."""
        return [self._columns[position] for position in self._input_positions()]

    @property
    def output_column(self) -> Hashable | None:
        """Column chosen as output, or ``None`` when none is chosen."""
        position = self._output_position()
        return None if position is None else self._columns[position]

    @property
    def selection(self) -> tuple[list[Hashable], Hashable | None]:
        """Return (input columns, output column)."""
        return self.input_columns, self.output_column

    @property
    def problem(self) -> str:
        """Return a specific live hint for the missing choice."""
        if self.output_column is None:
            return "Choose an output column (Y)."
        if not self.input_columns:
            return "Choose at least one input column (X), different from the output."
        return ""

    @property
    def output_chip_texts(self) -> list[str]:
        """Column names currently shown as the output chip."""
        return [chip.children[0].text for chip in self.output_chips.children]

    @property
    def input_chip_texts(self) -> list[str]:
        """Column names currently shown as input chips, in dataset order."""
        return [chip.children[0].text for chip in self.input_chips.children]

    def _input_positions(self) -> list[int]:
        """Positions of the columns checked as inputs."""
        return [position for position, check in enumerate(self._inputs) if check.checked]

    def _output_position(self) -> int | None:
        """Position of the column chosen as output, or ``None`` when none is chosen."""
        return next((position for position, radio in enumerate(self._outputs) if radio.checked), None)

    def _on_selection(self, event: ChangeEvent[bool]) -> None:
        """Recompute the mode and summary after any check box or radio button changes."""
        self._refresh_mode()
        self.edited.emit(UiEvent(source=self, origin=event.origin))

    def _refresh_mode(self) -> None:
        """Disable the Y column as an input and show the regression mode and the chosen columns as chips."""
        output = self._output_position()
        for position, check in enumerate(self._inputs):
            check.enabled = position != output
            if position == output and check.checked:
                check.checked = False

        inputs = self._input_positions()
        if not inputs or output is None:
            self.mode.text = "Choose at least one X and one Y"
            self.mode.foreground = self._tokens.muted
            self._show_chips(self.output_chips, [], primary=True)
            self._show_chips(self.input_chips, [], primary=False)
            self.summary_box.visible = False
            return

        self.mode.foreground = self._tokens.foreground
        self.mode.text = "Simple regression" if len(inputs) == 1 else "Multiple regression"
        input_texts = [self._texts[position] for position in inputs]
        self._show_chips(self.output_chips, [self._texts[output]], primary=True)
        self.input_caption.text = f"INPUTS (X) · {len(inputs)}"
        self._show_chips(self.input_chips, input_texts, primary=False)
        self.summary_box.visible = True

    def _show_chips(self, holder: Container, texts: list[str], primary: bool) -> None:
        """Replace the chips in ``holder``: the accent color marks the output, the panel color the inputs."""
        for chip in list(holder.children):
            chip.destroy()
        if primary:
            fill = self._tokens.accent
            text_color = self._tokens.accent_text
        else:
            fill = self._tokens.surface
            text_color = self._tokens.foreground
        for text in texts:
            chip = Container(parent=holder, layout="stack", padding=6, background=fill)
            Label(parent=chip, text=text, height=22, font_size=14, foreground=text_color, background=fill)


def _distinct_texts(columns: list[Hashable]) -> list[str]:
    """Return the text shown for each column, adding ``(#position)`` to texts that more than one column shares."""
    texts = [str(column) for column in columns]
    repeated = Counter(texts)
    return [f"{text} (#{position})" if repeated[text] > 1 else text for position, text in enumerate(texts, start=1)]
