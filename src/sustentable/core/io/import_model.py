"""Load a regression model saved by ``export_model``, without any GUI dependency.

A saved model is a single ``.onnx`` file: its graph computes the predictions, and its metadata (``metadata_props``,
one JSON value per key) holds the input and output columns, the coefficients and the intercept, R² and MSE on the
training and test subsets with their number of rows, and the user's description.

The model is rebuilt from that metadata with ``restore_model``, so it can be shown, used to predict and saved again
exactly like a model created in the application. Before it is returned, the graph is run once and compared with the
rebuilt model, so a file whose graph and metadata disagree is reported as damaged instead of being shown.
"""

import json
import math
import os
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

from sustentable.core.io.export_model import FORMAT_NAME, FORMAT_VERSION, MODEL_TYPE
from sustentable.core.model.regression import Metrics, RegressionModel, restore_model

# A saved model takes a few kilobytes. A file this large is not one, and reading it whole could exhaust the memory.
MAX_MODEL_BYTES = 64 * 1024 * 1024
# The graph computes in float32, so it only agrees with the float64 parameters up to that precision.
_GRAPH_TOLERANCE = 1e-4


class ModelImportError(Exception):
    """The model could not be loaded. The message is written for the user and names the file."""


@dataclass(frozen=True)
class SavedModel:
    """A model loaded from a file, with the description saved with it and the absolute path of the file."""

    model: RegressionModel
    description: str
    path: Path


def import_model(path: str | os.PathLike[str]) -> SavedModel:
    """Load the model saved in ``path`` by ``export_model``.

    Args:
        path: The ONNX file. The folder may contain spaces or any Unicode character, such as accents or ``ñ``.

    Returns:
        The rebuilt model, its description (empty when it was saved without one) and the absolute path of the file.

    Raises:
        ModelImportError: If the file cannot be read, is not a valid ONNX model, was not saved by this application,
            lacks part of the model information, or contains a graph that does not match that information.
    """
    file = Path(path).resolve()
    data = _read(file)
    try:
        onnx_model = onnx.load_model_from_string(data)
        onnx.checker.check_model(onnx_model)
    except Exception as error:  # protobuf and the ONNX checker raise many unrelated exception types.
        raise ModelImportError(f"File '{file}' is not a valid ONNX model. It may be damaged or have another format.") from error
    metadata = _Metadata(file, {prop.key: prop.value for prop in onnx_model.metadata_props})
    model, description = _rebuild(metadata)
    _check_graph(file, data, model)
    return SavedModel(model, description, file)


def _read(file: Path) -> bytes:
    """Read the whole file, refusing anything too large to be a saved model."""
    if not file.exists():
        raise ModelImportError(f"File '{file}' does not exist.")
    if not file.is_file():
        raise ModelImportError(f"'{file}' is not a file.")
    try:
        with file.open("rb") as stream:
            data = stream.read(MAX_MODEL_BYTES + 1)
    except PermissionError as error:
        raise ModelImportError(f"File '{file}' cannot be read: permission denied.") from error
    except OSError as error:
        raise ModelImportError(f"File '{file}' could not be read: {error.strerror or error}") from error
    if not data:
        raise ModelImportError(f"File '{file}' is empty.")
    if len(data) > MAX_MODEL_BYTES:
        raise ModelImportError(f"File '{file}' is too large to be a model saved by Sustentable.")
    return data


class _Metadata:
    """Decode the metadata of one file, reporting missing and invalid values with the name of the file."""

    def __init__(self, file: Path, entries: Mapping[str, str]) -> None:
        """Keep the raw JSON text of each entry; entries are only decoded when they are read."""
        self.file = file
        self._entries = entries

    def field(self, what: str, key: str, *path: str) -> object:
        """The value saved under ``key``, then under each key of ``path`` inside it; ``what`` names it in errors."""
        text = self._entries.get(key)
        if text is None:
            raise self.incomplete(what)
        try:
            value = json.loads(text, parse_constant=_reject_constant)
        except ValueError as error:
            raise self.damaged(what) from error
        for name in path:
            if not isinstance(value, dict):
                raise self.damaged(what)
            if name not in value:
                raise self.incomplete(what)
            value = value[name]
        return value

    def optional_text(self, key: str) -> str | None:
        """The text saved under ``key``, or ``None`` when it is missing or is not text."""
        try:
            value = self.field(key, key)
        except ModelImportError:
            return None
        return value if isinstance(value, str) else None

    def incomplete(self, what: str) -> ModelImportError:
        """The error for information that is missing from the file."""
        return ModelImportError(f"File '{self.file}' is incomplete: it does not contain the {what}.")

    def damaged(self, what: str) -> ModelImportError:
        """The error for information that the file contains but cannot be used."""
        return ModelImportError(f"File '{self.file}' is damaged: it has an invalid value for the {what}.")


def _rebuild(metadata: _Metadata) -> tuple[RegressionModel, str]:
    """Check that the file was saved by this application, then rebuild its model and read its description."""
    file = metadata.file
    if metadata.optional_text("format") != FORMAT_NAME or metadata.optional_text("model_type") != MODEL_TYPE:
        raise ModelImportError(f"File '{file}' is an ONNX model, but it was not saved by Sustentable as a linear regression.")
    version = _text(metadata, "format version", "version")
    if version.split(".")[0] != FORMAT_VERSION.split(".")[0]:
        raise ModelImportError(f"File '{file}' uses version {version} of the model format, which this version of Sustentable cannot load.")

    inputs = _labels(metadata, "input columns", "input_columns")
    outputs = _labels(metadata, "output column", "output_columns")
    if len(outputs) != 1:
        raise metadata.damaged("output column")
    coefficients = _numbers(metadata, "coefficients", "parameters", "coefficients")
    if len(coefficients) != len(inputs):
        raise metadata.damaged("coefficients")
    intercept = _number(metadata, "intercept", "parameters", "intercept")
    train = _metrics(metadata, "training", "train", "train_rows")
    test = _metrics(metadata, "test", "test", "test_rows")
    description = _text(metadata, "description", "description")
    try:
        model = restore_model(inputs, outputs[0], coefficients, intercept, train, test)
    except ValueError as error:
        raise metadata.damaged("input and output columns") from error
    return model, description


def _check_graph(file: Path, data: bytes, model: RegressionModel) -> None:
    """Run the graph on a row of zeros and on one row per input, and compare it with the rebuilt model.

    Those rows give the intercept and the intercept plus each coefficient, so any parameter that differs is found.
    """
    count = len(model.inputs)
    rows = np.vstack([np.zeros((1, count)), np.eye(count)])
    expected = model.estimator.predict(rows)
    mismatch = f"File '{file}' is damaged: the model it contains does not match the information saved with it."
    try:
        session = onnxruntime.InferenceSession(data, providers=["CPUExecutionProvider"])
        [graph_input] = session.get_inputs()
        actual = np.asarray(session.run(None, {graph_input.name: rows.astype(np.float32)})[0], dtype=float).reshape(-1)
    except Exception as error:  # ONNX Runtime raises many unrelated exception types for a graph it cannot run.
        raise ModelImportError(mismatch) from error
    tolerance = _GRAPH_TOLERANCE * (1 + abs(model.intercept) + rows @ np.abs(model.coefficients))
    if actual.shape != expected.shape or not np.all(np.abs(actual - expected) <= tolerance):
        raise ModelImportError(mismatch)


def _text(metadata: _Metadata, what: str, key: str) -> str:
    """Text saved under ``key``."""
    value = metadata.field(what, key)
    if not isinstance(value, str):
        raise metadata.damaged(what)
    return value


def _labels(metadata: _Metadata, what: str, key: str) -> list[Hashable]:
    """A non-empty list of column labels: text, numbers or booleans, as ``export_model`` writes them."""
    value = metadata.field(what, key)
    if not isinstance(value, list) or not value or not all(isinstance(label, str | int | float | bool) for label in value):
        raise metadata.damaged(what)
    return value


def _number(metadata: _Metadata, what: str, *keys: str) -> float:
    """A finite number saved under ``keys``."""
    number = _finite(metadata.field(what, *keys))
    if number is None:
        raise metadata.damaged(what)
    return number


def _numbers(metadata: _Metadata, what: str, *keys: str) -> list[float]:
    """A list of finite numbers saved under ``keys``."""
    value = metadata.field(what, *keys)
    numbers = [_finite(item) for item in value] if isinstance(value, list) else [None]
    if None in numbers:
        raise metadata.damaged(what)
    return [number for number in numbers if number is not None]


def _metrics(metadata: _Metadata, subset: str, key: str, rows_key: str) -> Metrics:
    """R², MSE and number of rows of one subset; a metric saved as ``null`` was undefined and becomes ``nan``."""
    values = []
    for name, metric in (("r2", "R²"), ("mse", "mean squared error")):
        what = f"{subset} {metric}"
        value = metadata.field(what, "metrics", key, name)
        number = math.nan if value is None else _finite(value)
        if number is None:
            raise metadata.damaged(what)
        values.append(number)
    what = f"number of {subset} rows"
    rows = metadata.field(what, "training", rows_key)
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise metadata.damaged(what)
    return Metrics(r2=values[0], mse=values[1], rows=rows)


def _finite(value: object) -> float | None:
    """``value`` as a float when it is a finite number, without accepting booleans; otherwise ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:  # JSON integers have no size limit.
        return None
    return number if math.isfinite(number) else None


def _reject_constant(name: str) -> float:
    """Refuse ``NaN`` and ``Infinity``, which Python's JSON reader accepts but standard JSON does not."""
    raise ValueError(f"{name} is not valid JSON.")
