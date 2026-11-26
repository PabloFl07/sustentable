"""Save a trained regression model as a single ONNX file, without any GUI dependency.

The format follows the model persistence research (``docs/model-saving-loading.md``), with the metadata stored inside
the model file instead of next to it:

- The ONNX graph holds the model, so predictions can be made with ONNX Runtime without the original dataset.
- The ONNX metadata (``metadata_props``, one JSON value per key) holds what the interface needs to show the model: the
  coefficients and the intercept, the formula, the input and output columns, R² and MSE on the training and test
  subsets, and the user's description.

The file contains neither the rows used to create the model nor the charts drawn for it. It is written to a temporary
file first and then moved into place, so a failed save never leaves a half-written file behind.
"""

import json
import math
import os
import platform
import tempfile
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from sustentable.core.model.regression import Metrics, RegressionModel

MODEL_EXTENSION = ".onnx"
FORMAT_NAME = "ONNX"
FORMAT_VERSION = "1.0"
MODEL_TYPE = "LinearRegression"
ONNX_INPUT_NAME = "input"


class ModelExportError(OSError):
    """The model could not be saved. The message is written for the user and names the file."""


@dataclass(frozen=True)
class ModelFiles:
    """Path of the file that holds a saved model."""

    model: Path

    @property
    def paths(self) -> tuple[Path, ...]:
        """Every file written for the model, which is only the ONNX file."""
        return (self.model,)


def model_files(path: str | os.PathLike[str]) -> ModelFiles:
    """Return the ONNX path for ``path``, adding the ``.onnx`` extension when it is missing.

    ``model`` becomes ``model.onnx``; ``model.ONNX`` keeps its own extension.

    Raises:
        ValueError: If ``path`` has no file name.
    """
    chosen = Path(path)
    if not chosen.name or chosen.name in (".", ".."):
        raise ValueError("Choose a file name to save the model.")
    if chosen.suffix.lower() != MODEL_EXTENSION:
        chosen = chosen.with_name(chosen.name + MODEL_EXTENSION)
    return ModelFiles(model=chosen)


def build_metadata(model: RegressionModel, description: str = "", model_file: str = "") -> dict[str, Any]:
    """Describe ``model`` with the metadata structure documented in the persistence research.

    Undefined metrics, such as R² over a single test row, are stored as ``null``. Only column names, parameters,
    metrics and row counts are included: never the values of the dataset.
    """
    return {
        "name": f"Linear regression - {model.output}",
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_type": MODEL_TYPE,
        "model_file": model_file,
        "formula": model.formula,
        "input_columns": [_column_label(column) for column in model.inputs],
        "output_columns": [_column_label(model.output)],
        "parameters": {"coefficients": list(model.coefficients), "intercept": model.intercept},
        "metrics": {"train": _metrics(model.train_metrics), "test": _metrics(model.test_metrics)},
        "description": description,
        "training": {"train_rows": model.train_metrics.rows, "test_rows": model.test_metrics.rows},
        "dependencies": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }


def export_model(model: RegressionModel, path: str | os.PathLike[str], description: str = "") -> ModelFiles:
    """Save ``model`` and its metadata as ``<name>.onnx``, replacing an existing file.

    Args:
        model: The trained model to save.
        path: Destination of the ONNX file. The ``.onnx`` extension is added when it is missing. The folder may
            contain spaces or any Unicode character, such as accents or ``ñ``.
        description: Text written by the user. It may be empty.

    Returns:
        The path of the saved file.

    Raises:
        ModelExportError: If the folder does not exist, cannot be written, or the model cannot be converted.
            An existing file is left untouched when this happens.
    """
    files = model_files(path)
    metadata = build_metadata(model, description, files.model.name)
    try:
        onnx_bytes = _onnx_bytes(model, metadata)
    except Exception as error:  # skl2onnx and onnx raise many unrelated exception types.
        raise ModelExportError(f"The model could not be converted to {FORMAT_NAME}: {error}") from error

    folder = files.model.parent
    if not folder.is_dir():
        raise ModelExportError(f"The folder '{folder}' does not exist. Choose another folder to save the model.")
    temporary: list[Path] = []
    try:
        for data in (onnx_bytes,):
            temporary.append(_write_temporary(folder, data))
        for source, target in zip(temporary, files.paths, strict=True):
            os.replace(source, target)
    except PermissionError as error:
        raise ModelExportError(f"There is no permission to save the model in '{folder}'. Choose another folder.") from error
    except OSError as error:
        reason = error.strerror or str(error)
        raise ModelExportError(f"The model could not be saved to '{files.model}': {reason}") from error
    finally:
        for leftover in temporary:
            leftover.unlink(missing_ok=True)
    return files


def _onnx_bytes(model: RegressionModel, metadata: dict[str, Any]) -> bytes:
    """Convert the estimator to a validated ONNX model that takes a float32 matrix with one column per input.

    Each metadata entry is stored as one ``metadata_props`` key whose value is the entry written as JSON.
    """
    initial_types = [(ONNX_INPUT_NAME, FloatTensorType([None, len(model.inputs)]))]
    onnx_model = convert_sklearn(model.estimator, initial_types=initial_types)
    onnx_model.doc_string = model.formula
    onnx.helper.set_model_props(onnx_model, {key: json.dumps(value, ensure_ascii=False) for key, value in metadata.items()})
    onnx.checker.check_model(onnx_model)
    return onnx_model.SerializeToString()


def _write_temporary(folder: Path, data: bytes) -> Path:
    """Write ``data`` to a new hidden temporary file in ``folder`` and return its path."""
    descriptor, name = tempfile.mkstemp(dir=folder, prefix=".sustentable-", suffix=".tmp")
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _metrics(metrics: Metrics) -> dict[str, float | None]:
    """R² and MSE of one subset, with undefined values as ``None``."""
    return {"r2": _finite(metrics.r2), "mse": _finite(metrics.mse)}


def _finite(value: float) -> float | None:
    """Keep finite values; JSON has no standard representation for ``nan`` or infinity."""
    return float(value) if math.isfinite(value) else None


def _column_label(label: Hashable) -> str | int | float | bool:
    """Keep column labels JSON can represent; write any other label as text."""
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, str | int | float | bool):
        return label
    return str(label)
