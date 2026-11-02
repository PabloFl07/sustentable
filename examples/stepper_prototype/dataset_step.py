"""Dataset step content, shared by the standalone and grouped layouts."""

from pysual import Button, ClickEvent, Container, Label


class DatasetForm(Container):
    """Simulate choosing a dataset with one click."""

    def _initialize(self) -> None:
        super()._initialize()
        self.layout, self.spacing, self.height = "stack", 8, 64
        self.loaded = False
        self.filename = Label(parent=self, text="No sample selected", height=24)
        self.sample = Button(parent=self, text="Use sample dataset", height=32, width=176)
        self.sample.click.connect(self._load)
        self.editors = (self.sample,)

    def _load(self, event: ClickEvent) -> None:
        self.loaded = True
        self.filename.text = "sample.csv · 6 rows"
        self.sample.update(text="Sample loaded", enabled=False)

    @property
    def problem(self) -> str:
        """Return the sample validation message."""
        return "" if self.loaded else "Choose the sample dataset."
