"""Predict the output of a model for one set of input values typed by the user, without any GUI dependency.

The same functions work with any model that exposes its ``inputs``, its ``output`` and a ``predict`` method that
takes a frame with the input columns, so a model created in the application and a model loaded from a file give
the same result for the same values.
"""

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from sustentable.core.model.regression import DECIMALS


class Predictor(Protocol):
    """A model that predicts ``output`` from the ``inputs`` columns of each row of a frame."""

    @property
    def inputs(self) -> tuple[Hashable, ...]:
        """Labels of the input columns, in the order the model uses them."""
        ...

    @property
    def output(self) -> Hashable:
        """Label of the output column."""
        ...

    def predict(self, frame: DataFrame) -> NDArray[np.float64]:
        """Predict the output for each row of ``frame``."""
        ...


class PredictionInputError(ValueError):
    """Some input values are missing or are not numbers. The message is written for the user and names them.

    ``missing`` and ``invalid`` hold the labels of the affected input columns, in the order of the model's inputs.
    """

    def __init__(self, missing: Sequence[Hashable], invalid: Sequence[Hashable]) -> None:
        """Keep the affected columns and build a message that names each of them."""
        self.missing = tuple(missing)
        self.invalid = tuple(invalid)
        problems = []
        if self.missing:
            problems.append(f"Enter a value for {_names(self.missing)}.")
        if self.invalid:
            noun = "is not a valid number" if len(self.invalid) == 1 else "are not valid numbers"
            problems.append(f"The value of {_names(self.invalid)} {noun}.")
        super().__init__(" ".join(problems))


def parse_number(text: str) -> float:
    """Read a decimal number written with a point or a comma as the decimal separator, such as ``3.5`` or ``3,5``.

    Surrounding blank space is ignored. Thousands separators are not accepted, so ``1,234.5`` is rejected instead of
    being read as a different number than the user meant.

    Raises:
        ValueError: If ``text`` is empty or is not a finite number.
    """
    cleaned = text.strip()
    if "," in cleaned:
        if "." in cleaned or cleaned.count(",") > 1:
            raise ValueError(f"'{text}' is not a valid number.")
        cleaned = cleaned.replace(",", ".")
    # float() also accepts digit separators such as "1_000", which are not a way users write numbers.
    if not cleaned or "_" in cleaned:
        raise ValueError(f"'{text}' is not a valid number.")
    try:
        value = float(cleaned)
    except ValueError as error:
        raise ValueError(f"'{text}' is not a valid number.") from error
    if not math.isfinite(value):
        raise ValueError(f"'{text}' is not a valid number.")
    return value


def predict_value(model: Predictor, values: Mapping[Hashable, str | float]) -> float:
    """Predict the output of ``model`` for one value of each of its input columns.

    Args:
        model: A created or loaded model.
        values: Value of each input column, keyed by the column label. Text is read with ``parse_number``, so both
            ``3.5`` and ``3,5`` are accepted. Extra keys are ignored.

    Returns:
        The predicted output value.

    Raises:
        PredictionInputError: If a value is missing, blank, or is not a finite number. The message names every
            affected column.
    """
    missing: list[Hashable] = []
    invalid: list[Hashable] = []
    row: list[float] = []
    for column in model.inputs:
        value = values.get(column)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(column)
            continue
        try:
            number = parse_number(value) if isinstance(value, str) else float(value)
        except TypeError, ValueError:
            invalid.append(column)
            continue
        if not math.isfinite(number):
            invalid.append(column)
            continue
        row.append(number)
    if missing or invalid:
        raise PredictionInputError(missing, invalid)
    frame = DataFrame([row], columns=list(model.inputs))
    return float(model.predict(frame)[0])


def format_prediction(output: Hashable, value: float, decimals: int = DECIMALS) -> str:
    """Write ``<output> = <value>``, rounded to the same decimals as the model's formula."""
    text = f"{value:.{decimals}f}"
    # A tiny negative value would otherwise be shown as "-0.0000".
    if float(text) == 0:
        text = text.removeprefix("-")
    return f"{output} = {text}"


def _names(columns: Sequence[Hashable]) -> str:
    """Quote the column labels and join them as ``'a'``, ``'a' and 'b'`` or ``'a', 'b' and 'c'``."""
    quoted = [f"'{column}'" for column in columns]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"
