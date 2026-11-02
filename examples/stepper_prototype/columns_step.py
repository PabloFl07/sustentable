"""Column selection content and its small demonstration validator."""

from pysual import CheckBox, Container, Dropdown, Label


class ColumnsForm(Container):
    """Provide a repeatable selection to compare validation styles."""

    def _initialize(self) -> None:
        super()._initialize()
        self.layout, self.spacing, self.height = "stack", 8, 96
        row = Container(parent=self, layout="stack", direction="horizontal", spacing=8, height=32)
        Label(parent=row, text="Target (Y)", width=80)
        self.target = Dropdown(parent=row, items=("Energy", "Temperature"), placeholder="Choose target", flex=1)
        Label(parent=self, text="Inputs (X)", height=24)
        inputs = Container(parent=self, layout="stack", direction="horizontal", spacing=8, height=24)
        self.area = CheckBox(parent=inputs, text="Area", flex=1)
        self.occupants = CheckBox(parent=inputs, text="Occupants", flex=1)
        self.editors = (self.target, self.area, self.occupants)

    @property
    def problem(self) -> str:
        """Require a target and one input in validation demos."""
        if self.target.selected_index < 0:
            return "Choose a target (Y)."
        return "" if self.area.checked or self.occupants.checked else "Choose at least one input (X)."
