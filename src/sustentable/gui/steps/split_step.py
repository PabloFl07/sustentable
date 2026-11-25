"""Lightweight allocation preview with an explicit asynchronous train/test split."""

import asyncio
from typing import ClassVar

from pandas import DataFrame
from pysual import ChangeEvent, Container, Event, Label, NumericEditor, NumericInput, Slider, Step, UiEvent

from sustentable.core.processing.data_split import count_test_rows, split_dataset

DEFAULT_TEST_PERCENT = 20
# Default seed, so the same dataset and percentage always produce the same subsets unless the user picks another.
SPLIT_SEED = 42
# scikit-learn accepts integer seeds in [0, 2**32 - 1].
MAX_SEED = 2**32 - 1
_NO_DATASET = "Confirm the columns before splitting."


class _SeedEditor(NumericEditor):
    """Numeric editor popup without pysual's default keyboard hint."""

    hint: ClassVar[str] = ""


class _SeedInput(NumericInput):
    """Numeric input whose editor popup shows only validation messages."""

    def _create_editor_content(self) -> NumericEditor:
        return _SeedEditor(self.value, decimals=self.decimals)


class SplitStep(Step):
    """Preview the allocation and leave confirmation to the window's common footer."""

    edited: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, **properties: object) -> None:
        """Create compact percentage and row-count controls."""
        super().__init__(title="Split", padding=8, spacing=8, **properties)
        self._frame: DataFrame | None = None
        row = Container(parent=self, layout="stack", direction="horizontal", spacing=16, height=32)
        self.allocation = Label(parent=row, width=220)
        self.share = Slider(parent=row, minimum=0, maximum=100, value=DEFAULT_TEST_PERCENT, flex=1)
        seed_row = Container(parent=self, layout="stack", direction="horizontal", spacing=16, height=32)
        Label(parent=seed_row, text="Random seed", width=220)
        self.seed_input = _SeedInput(parent=seed_row, minimum=0, maximum=MAX_SEED, value=SPLIT_SEED, decimals=0, width=180)
        self.counts = Label(parent=self, height=24)
        self._refresh()

    def reset(self, frame: DataFrame | None) -> None:
        """Restore the default allocation when a dataset or confirmed selection changes."""
        self._frame = frame
        self.share.value = DEFAULT_TEST_PERCENT
        self._refresh()

    @property
    def problem(self) -> str:
        """Explain why the current allocation cannot produce two usable subsets."""
        if self._frame is None:
            return _NO_DATASET
        train, test = self.row_counts
        if train + test < 2:
            return "At least two rows are needed for training and test."
        if not train or not test:
            return "Adjust the percentage so both training and test contain rows."
        return ""

    @property
    def seed(self) -> int:
        """Return the seed the next confirmed split uses, so the same choice always reproduces the same subsets."""
        return int(self.seed_input.value)

    @property
    def row_counts(self) -> tuple[int, int]:
        """Match the backend's rounding without shuffling or copying any rows."""
        total = 0 if self._frame is None else len(self._frame)
        test = count_test_rows(total, self.share.value / 100)
        return total - test, test

    async def split(self) -> tuple[DataFrame, DataFrame]:
        """Create the confirmed subsets once, off the UI thread, using the selected seed."""
        if problem := self.problem:
            raise ValueError(problem)
        return await asyncio.to_thread(split_dataset, self._frame, self.share.value / 100, self.seed)

    def share_on_changed(self, event: ChangeEvent[float]) -> None:
        """Update cheap live estimates; actual data processing belongs to confirmation."""
        self._refresh()
        self.edited.emit(UiEvent(source=self, origin=event.origin))

    def seed_input_on_changed(self, event: ChangeEvent[float]) -> None:
        """Invalidate the confirmed subsets, since another seed shuffles the rows differently."""
        self.edited.emit(UiEvent(source=self, origin=event.origin))

    def _refresh(self) -> None:
        self.allocation.text = f"Train: {100 - self.share.value:.1f}% | Test: {self.share.value:.1f}%"
        if self._frame is None:
            self.counts.text = _NO_DATASET
        else:
            train, test = self.row_counts
            self.counts.text = f"Training: {train:,} rows · Test: {test:,} rows"
