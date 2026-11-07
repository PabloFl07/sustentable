"""Missing-value report and treatment options for the preprocessing step."""

from collections.abc import Hashable
from typing import ClassVar

from pandas import DataFrame
from pysual import Button, ChangeEvent, ClickEvent, Container, Dropdown, Event, Label, ScrollArea, Step, TextBox, UiEvent

from sustentable.core.processing.data_processing import NanStrategy, PreprocessResult, detect_nan_values, parse_constant

STRATEGY_LABELS: dict[NanStrategy, str] = {
    NanStrategy.DELETE: "Delete rows with missing values",
    NanStrategy.MEAN: "Fill with the column mean",
    NanStrategy.MEDIAN: "Fill with the column median",
    NanStrategy.CONSTANT: "Fill with a constant value",
}


class PreprocessStep(Step):
    """Report missing values per column and subset, and collect the strategy to treat them.

    The owner window runs the treatment when ``apply`` is emitted and confirms the step.
    ``edited`` is emitted whenever an option changes, so an applied result can be discarded.
    After a result is shown, the counters report the values still missing next to the original ones.
    """

    apply: ClassVar[Event[UiEvent]] = Event(UiEvent)
    edited: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, **properties: object) -> None:
        """Place the missing-value report next to the treatment options."""
        super().__init__(title="Preprocess", direction="horizontal", height=196, padding=8, spacing=16, **properties)
        self._report: DataFrame | None = None
        self._current: DataFrame | None = None
        self._rows: list[Label] = []

        report = Container(parent=self, layout="stack", spacing=4, flex=1)
        self.summary = Label(parent=report, text="Confirm the split before preprocessing.", height=24)
        self.report_list = ScrollArea(parent=report, layout="stack", spacing=4, flex=1)

        options = Container(parent=self, layout="stack", spacing=8, flex=1)
        strategy_row = Container(parent=options, layout="stack", direction="horizontal", spacing=12, height=36)
        Label(parent=strategy_row, text="Strategy", width=112)
        self.strategy = Dropdown(parent=strategy_row, items=tuple(STRATEGY_LABELS.values()), selected_index=0, flex=1)
        constant_row = Container(parent=options, layout="stack", direction="horizontal", spacing=12, height=36)
        Label(parent=constant_row, text="Constant value", width=112)
        self.constant = TextBox(parent=constant_row, placeholder="Numeric value, e.g. 0", flex=1)
        self.apply_button = Button(parent=options, text="Apply preprocessing", width=200, height=36)
        self.status = Label(parent=options, text="", height=24, word_wrap=True)
        self._refresh()

    def load(self, train: DataFrame | None, test: DataFrame | None, columns: list[Hashable]) -> None:
        """Report the missing values of a confirmed split, or clear the report when ``train`` is ``None``."""
        self._report = None if train is None or test is None else detect_nan_values(train, test, columns)
        self._current = None
        self.status.text = ""
        for row in self._rows:
            row.destroy()
        self._rows = []
        if self._report is not None:
            self._rows = [Label(parent=self.report_list, height=22) for _ in range(len(self._report))]
        self._refresh()

    @property
    def has_missing_values(self) -> bool:
        """Whether the reported split has any missing value in the selected columns."""
        return self._report is not None and bool(self._report[["train", "test"]].to_numpy().any())

    @property
    def selected_strategy(self) -> NanStrategy:
        """Strategy chosen in the dropdown."""
        return list(STRATEGY_LABELS)[max(self.strategy.selected_index, 0)]

    @property
    def constant_value(self) -> float | None:
        """Parsed constant for the constant strategy, or ``None`` for the other strategies.

        Raises:
            ValueError: If the constant strategy is chosen and the text is not a finite number.
        """
        return parse_constant(self.constant.text) if self.selected_strategy is NanStrategy.CONSTANT else None

    @property
    def problem(self) -> str:
        """Explain what prevents continuing to the model."""
        if self._report is None:
            return "Confirm the split before preprocessing."
        return ""

    def show_result(self, result: PreprocessResult) -> None:
        """Describe the outcome of an applied strategy for each subset."""
        noun = "rows removed" if result.strategy is NanStrategy.DELETE else "values filled"
        if self._report is not None:
            self._current = detect_nan_values(result.train, result.test, self._report.column)
        self._refresh()
        self.status.text = f"Preprocessing applied. Training: {result.train_changes:,} {noun} · Test: {result.test_changes:,} {noun}."

    def clear_result(self) -> None:
        """Show the missing values of the split again after an applied result is discarded."""
        if self._current is not None:
            self._current = None
            self.status.text = ""
            self._refresh()

    @property
    def remaining_missing_values(self) -> tuple[int, int]:
        """Missing values shown by the counters for the training and test subsets."""
        report = self._current if self._current is not None else self._report
        return (0, 0) if report is None else (int(report.train.sum()), int(report.test.sum()))

    def strategy_on_changed(self, event: ChangeEvent[int]) -> None:
        """Enable the constant field only for the constant strategy."""
        self._options_edited(event)

    def constant_on_changed(self, event: ChangeEvent[str]) -> None:
        """Discard an applied result after the constant changes."""
        self._options_edited(event)

    def apply_button_on_click(self, event: ClickEvent) -> None:
        """Ask the owning window to apply the chosen strategy."""
        self.apply.emit(UiEvent(source=self, origin=event.origin))

    def _options_edited(self, event: UiEvent) -> None:
        self.status.text = ""
        if self.has_missing_values:
            self._current = None
        self._refresh()
        self.edited.emit(UiEvent(source=self, origin=event.origin))

    def _refresh(self) -> None:
        self.constant.enabled = self.selected_strategy is NanStrategy.CONSTANT
        self.apply_button.enabled = self.has_missing_values
        if self._report is None:
            self.summary.text = "Confirm the split before preprocessing."
            return
        for row, original, current in zip(self._rows, self._report.itertuples(index=False), self._counts(), strict=True):
            row.text = f"{original.column}: training {_count(current.train, original.train)} · test {_count(current.test, original.test)}"
        train, test = self.remaining_missing_values
        if not self.has_missing_values:
            self.summary.text = "No missing values in the selected columns. You can continue."
        elif self._current is not None:
            self.summary.text = f"Missing values after preprocessing: training {train:,} · test {test:,}"
        else:
            self.summary.text = f"Missing values: training {train:,} · test {test:,}"

    def _counts(self) -> list:
        """Rows of the counters currently shown: the processed result when there is one, otherwise the split."""
        report = self._current if self._current is not None else self._report
        return list(report.itertuples(index=False))


def _count(current: int, original: int) -> str:
    """Format a counter, keeping the original count visible when it changed."""
    return f"{current:,}" if current == original else f"{current:,} (was {original:,})"
