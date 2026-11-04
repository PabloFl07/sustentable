"""Lightweight allocation preview with an explicit asynchronous train/test split."""

import asyncio
from typing import ClassVar

from pandas import DataFrame
from pysual import ChangeEvent, Container, Event, Label, Slider, Step, UiEvent

from sustentable.core.data_shufler import data_shuffler


class SplitStep(Step):
    """Preview the allocation and leave confirmation to the window's common footer."""

    edited: ClassVar[Event[UiEvent]] = Event(UiEvent)

    def __init__(self, **properties: object) -> None:
        """Create compact percentage and row-count controls."""
        super().__init__(title="Split", padding=8, spacing=8, **properties)
        self._frame: DataFrame | None = None
        row = Container(parent=self, layout="stack", direction="horizontal", spacing=16, height=32)
        self.allocation = Label(parent=row, text="Train: 80.0% | Test: 20.0%", width=220)
        self.share = Slider(parent=row, minimum=0, maximum=100, value=20, flex=1)
        self.counts = Label(parent=self, text="Confirm the columns before splitting.", height=24)

    def reset(self, frame: DataFrame | None) -> None:
        """Restore the default allocation when a dataset or confirmed selection changes."""
        self._frame = frame
        self.share.value = 20
        self._refresh()

    @property
    def problem(self) -> str:
        """Explain why the current allocation cannot produce two usable subsets."""
        if self._frame is None:
            return "Confirm the columns before splitting."
        train, test = self.row_counts
        if train + test < 2:
            return "At least two rows are needed for training and test."
        if not train or not test:
            return "Adjust the percentage so both training and test contain rows."
        return ""

    @property
    def row_counts(self) -> tuple[int, int]:
        """Match the backend's rounding without shuffling or copying any rows."""
        total = 0 if self._frame is None else len(self._frame)
        test = round(total * (self.share.value / 100))
        return total - test, test

    async def split(self) -> tuple[DataFrame, DataFrame]:
        """Create the confirmed subsets once, off the UI thread, using the existing seed."""
        if problem := self.problem:
            raise ValueError(problem)
        test, train = await asyncio.to_thread(data_shuffler, self._frame, seed=42, test_percent=float(self.share.value / 100))
        return train, test

    def share_on_changed(self, event: ChangeEvent[float]) -> None:
        """Update cheap live estimates; actual data processing belongs to confirmation."""
        self._refresh()
        self.edited.emit(UiEvent(source=self, origin=event.origin))

    def _refresh(self) -> None:
        self.allocation.text = f"Train: {100 - self.share.value:.1f}% | Test: {self.share.value:.1f}%"
        if self._frame is None:
            self.counts.text = "Confirm the columns before splitting."
        else:
            train, test = self.row_counts
            self.counts.text = f"Training: {train:,} rows · Test: {test:,} rows"
