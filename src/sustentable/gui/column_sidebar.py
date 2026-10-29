"""Side panel where the user picks the input (X) and output (Y) columns of a regression."""

from collections import Counter
from collections.abc import Hashable, Iterable
from itertools import count

from pysual import Button, ChangeEvent, CheckBox, Container, Label, RadioButton, ScrollArea, Tokens

from sustentable.gui.controls.sidebar_button import SidebarButton


class ColumnSidebar(ScrollArea):
    """List the dataset columns as X check boxes and Y radio buttons, and summarize the chosen columns.

    The owner window wires ``hide_panel`` and ``confirm``; this panel only keeps the selection consistent.
    A column chosen as Y cannot also be an X. Column labels are kept as they are in the DataFrame,
    so the selection can index the frame directly.

    Controls are kept by column position, so repeated labels each get their own row. When two labels print
    the same, the shown text adds the column's 1-based position so the user can tell them apart.
    """

    _instances = count()

    def __init__(self, tokens: Tokens, **properties: object) -> None:
        """Create the panel colored with the window theme's ``tokens``; it starts with no columns."""
        defaults: dict[str, object] = {"layout": "stack", "width": 280, "padding": 8, "spacing": 6, "background": tokens.surface}
        defaults.update(properties)
        super().__init__(**defaults)
        self._background = tokens.surface
        self._tokens = tokens
        # Each panel gets its own radio group so two panels never share the Y choice.
        self._output_group_name = f"column-sidebar-output-{next(self._instances)}"
        self._columns: list[Hashable] = []
        self._texts: list[str] = []
        self._inputs: list[CheckBox] = []
        self._outputs: list[RadioButton] = []
        background = self._background

        self.header = Container(parent=self, layout="stack", direction="horizontal", height=28)
        Label(parent=self.header, text="COLUMNS", font_size=18, flex=1, background=background)
        self.hide_panel = SidebarButton(parent=self.header, icon="chevron_left", width=28, height=28, tooltip="Hide panel")

        Label(parent=self, text="Input (X)", height=22, font_size=16, background=background)
        self.input_group = Container(parent=self, layout="stack", spacing=4)
        Label(parent=self, text="Output (Y)", height=22, font_size=16, background=background)
        self.output_group = Container(parent=self, layout="stack", spacing=4)

        # Mode header (outside the box)
        self.mode = Label(parent=self, text="", height=24, font_size=16, background=background)
        # Box summarizing the selection: the Y column, the X columns as chips, and the formula
        box = tokens.raised
        self.formula_box = Container(parent=self, layout="stack", padding=10, spacing=6, background=box)
        Label(parent=self.formula_box, text="OUTPUT (Y)", height=18, font_size=11, foreground=tokens.muted, background=box)
        self.output_chips = Container(parent=self.formula_box, layout="flow", direction="horizontal", spacing=6)
        self.input_caption = Label(parent=self.formula_box, text="INPUTS (X)", height=18, font_size=13, foreground=tokens.muted, background=box)
        self.input_chips = Container(parent=self.formula_box, layout="flow", direction="horizontal", spacing=6)
        Container(parent=self.formula_box, height=1, background=tokens.border)
        self.formula = Label(parent=self.formula_box, text="", font_size=13, word_wrap=True, foreground=tokens.muted, background=box)

        self.confirm = Button(parent=self, text="Confirm selection", icon="check", height=30, enabled=False)
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
            check = CheckBox(parent=self.input_group, text=text, height=24, background=self._background, font_size=14)
            check.changed.connect(self._on_selection)
            self._inputs.append(check)

            radio = RadioButton(
                parent=self.output_group, text=text, group=self._output_group_name, height=24, background=self._background, font_size=14
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
        """Recompute the mode and formula after any check box or radio button changes."""
        self._refresh_mode()

    def _refresh_mode(self) -> None:
        """Disable the Y column as an input and show the regression mode, formula, and confirm state."""
        output = self._output_position()
        for position, check in enumerate(self._inputs):
            check.enabled = position != output
            if position == output and check.checked:
                check.checked = False

        inputs = self._input_positions()
        if not inputs or output is None:
            self.mode.text = "Choose at least one X and one Y"
            self.formula.text = ""
            self._show_chips(self.output_chips, [], primary=True)
            self._show_chips(self.input_chips, [], primary=False)
            self.formula_box.visible = False
            self.confirm.enabled = False
            return

        if len(inputs) == 1:
            self.mode.text = "Simple regression"
        else:
            self.mode.text = "Multiple regression"
        input_texts = [self._texts[position] for position in inputs]
        self._show_chips(self.output_chips, [self._texts[output]], primary=True)
        self.input_caption.text = "INPUTS (X) · " + str(len(inputs))
        self._show_chips(self.input_chips, input_texts, primary=False)
        self.formula.text = self._texts[output] + " ~ " + " + ".join(input_texts)
        self.formula_box.visible = True
        self.confirm.enabled = True

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
            chip = Container(parent=holder, layout="stack", padding=4, background=fill)
            Label(parent=chip, text=text, font_size=13, foreground=text_color, background=fill)


def _distinct_texts(columns: list[Hashable]) -> list[str]:
    """Return the text shown for each column, adding ``(#position)`` to texts that more than one column shares."""
    texts = [str(column) for column in columns]
    repeated = Counter(texts)
    return [f"{text} (#{position})" if repeated[text] > 1 else text for position, text in enumerate(texts, start=1)]
