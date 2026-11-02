"""Split step content; the slider never transforms the sample data."""

from pysual import ChangeEvent, Container, Label, Slider


class SplitForm(Container):
    """Show a split control without splitting or transforming any data."""

    def _initialize(self) -> None:
        super()._initialize()
        self.layout, self.spacing, self.height = "stack", 8, 96
        self.allocation = Label(parent=self, text="80% training · 20% test", height=24)
        self.share = Slider(parent=self, minimum=0, maximum=100, value=20, height=32, tooltip="Test percentage")
        Label(parent=self, text="Preview only · no split is performed", height=24)
        self.share.changed.connect(self._update)
        self.editors = (self.share,)

    def _update(self, event: ChangeEvent[float]) -> None:
        self.allocation.text = f"{100 - self.share.value:g}% training · {self.share.value:g}% test"

    @property
    def problem(self) -> str:
        """Make an invalid boundary easy to try in the validation demos."""
        return "" if 0 < self.share.value < 100 else "Leave some rows in both sets."
