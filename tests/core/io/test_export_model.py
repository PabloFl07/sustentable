"""Unit tests for saving a regression model as ONNX plus JSON metadata, without the dataset or the charts."""

import json
import math
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import numpy_helper
from pandas import DataFrame

from sustentable.core.io import export_model as export_module
from sustentable.core.io.export_model import ModelExportError, build_metadata, export_model, model_files
from sustentable.core.model.regression import RegressionModel, create_model

# Values that only appear in the dataset, so finding them in a saved file would mean the data leaked into it.
TRAIN = DataFrame(
    {
        "temperatura": [11.125, 13.375, 17.625, 19.875, 23.125, 29.375],
        "humedad": [41.5, 47.25, 52.75, 49.5, 63.25, 58.75],
        "consumo": [101.0625, 117.4375, 139.8125, 142.1875, 171.5625, 189.9375],
    }
)
TEST = DataFrame({"temperatura": [15.625, 21.875], "humedad": [44.75, 61.25], "consumo": [126.3125, 163.6875]})
EXPECTED_KEYS = {
    "name",
    "format",
    "version",
    "model_type",
    "model_file",
    "formula",
    "input_columns",
    "output_columns",
    "parameters",
    "metrics",
    "description",
    "training",
    "dependencies",
}


@pytest.fixture
def model() -> RegressionModel:
    """A multiple regression trained on the small dataset above."""
    return create_model(TRAIN, TEST, ["temperatura", "humedad"], "consumo")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_values() -> set[float]:
    return {float(value) for frame in (TRAIN, TEST) for value in frame.to_numpy().ravel()}


# ----------------------------------------------------------------------------
# File names
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chosen", "model_name", "metadata_name"),
    [
        ("energía", "energía.onnx", "energía_metadata.json"),
        ("energía.onnx", "energía.onnx", "energía_metadata.json"),
        ("modelo.ONNX", "modelo.ONNX", "modelo_metadata.json"),
        ("modelo.v2", "modelo.v2.onnx", "modelo.v2_metadata.json"),
    ],
)
def test_extension_is_added_when_missing(tmp_path: Path, chosen: str, model_name: str, metadata_name: str) -> None:
    """The ONNX extension is added automatically, and the metadata file sits next to the model."""
    files = model_files(tmp_path / chosen)
    assert files.model == tmp_path / model_name
    assert files.metadata == tmp_path / metadata_name


# ----------------------------------------------------------------------------
# Saved information
# ----------------------------------------------------------------------------


def test_metadata_contains_everything_needed_to_show_the_model(tmp_path: Path, model: RegressionModel) -> None:
    """Coefficients, intercept, formula, columns, R² and MSE on both subsets, and the description are saved."""
    files = export_model(model, tmp_path / "modelo", "Predice el consumo según la temperatura y la humedad.")
    metadata = _load(files.metadata)

    assert set(metadata) == EXPECTED_KEYS
    assert metadata["format"] == "ONNX" and metadata["model_type"] == "LinearRegression"
    assert metadata["model_file"] == "modelo.onnx"
    assert metadata["formula"] == model.formula
    assert metadata["formula"].startswith("consumo = ") and "temperatura" in metadata["formula"] and "humedad" in metadata["formula"]
    assert metadata["input_columns"] == ["temperatura", "humedad"]
    assert metadata["output_columns"] == ["consumo"]
    assert metadata["parameters"]["coefficients"] == pytest.approx(list(model.coefficients))
    assert metadata["parameters"]["intercept"] == pytest.approx(model.intercept)
    assert metadata["metrics"] == {
        "train": {"r2": pytest.approx(model.train_metrics.r2), "mse": pytest.approx(model.train_metrics.mse)},
        "test": {"r2": pytest.approx(model.test_metrics.r2), "mse": pytest.approx(model.test_metrics.mse)},
    }
    assert metadata["description"] == "Predice el consumo según la temperatura y la humedad."
    assert metadata["training"] == {"train_rows": 6, "test_rows": 2}


def test_empty_description_is_saved(tmp_path: Path, model: RegressionModel) -> None:
    """The description is optional."""
    files = export_model(model, tmp_path / "modelo.onnx")
    assert _load(files.metadata)["description"] == ""


def test_undefined_metric_is_saved_as_null(tmp_path: Path) -> None:
    """R² over a single test row is not defined, and JSON stores it as null instead of a non-standard NaN."""
    model = create_model(TRAIN, TEST.head(1), ["temperatura"], "consumo")
    assert math.isnan(model.test_metrics.r2)
    files = export_model(model, tmp_path / "modelo.onnx")
    assert "NaN" not in files.metadata.read_text(encoding="utf-8")
    assert _load(files.metadata)["metrics"]["test"]["r2"] is None


def test_onnx_model_predicts_without_the_dataset(tmp_path: Path, model: RegressionModel) -> None:
    """The saved ONNX model gives the same predictions as the trained model, using only the new rows."""
    files = export_model(model, tmp_path / "modelo.onnx")
    session = onnxruntime.InferenceSession(files.model.read_bytes(), providers=["CPUExecutionProvider"])
    rows = DataFrame({"temperatura": [5.0, 40.0], "humedad": [30.0, 90.0]})
    predicted = session.run(None, {session.get_inputs()[0].name: rows.to_numpy(dtype=np.float32)})[0].reshape(-1)
    np.testing.assert_allclose(predicted, model.predict(rows), rtol=1e-4)


# ----------------------------------------------------------------------------
# Information that is not saved
# ----------------------------------------------------------------------------


def test_metadata_does_not_contain_the_dataset(tmp_path: Path, model: RegressionModel) -> None:
    """No value of the dataset, nor its training or test subsets, nor any chart is written to the metadata."""
    files = export_model(model, tmp_path / "modelo.onnx")
    metadata = _load(files.metadata)

    def numbers(value: object) -> list[float]:
        if isinstance(value, dict):
            return [number for item in value.values() for number in numbers(item)]
        if isinstance(value, list):
            return [number for item in value for number in numbers(item)]
        return [float(value)] if isinstance(value, int | float) and not isinstance(value, bool) else []

    assert not _dataset_values() & set(numbers(metadata))
    assert set(metadata) == EXPECTED_KEYS
    text = files.metadata.read_text(encoding="utf-8").lower()
    for forbidden in ("dataset", "points", "chart", "plot", "image", "png"):
        assert forbidden not in text
    assert {path.name for path in tmp_path.iterdir()} == {"modelo.onnx", "modelo_metadata.json"}


def test_onnx_file_only_stores_the_parameters(tmp_path: Path, model: RegressionModel) -> None:
    """The ONNX graph keeps only the coefficients and the intercept, not the rows used to train it."""
    files = export_model(model, tmp_path / "modelo.onnx")
    graph = onnx.load_from_string(files.model.read_bytes()).graph
    tensors = [*graph.initializer, *(attribute.t for node in graph.node for attribute in node.attribute if attribute.HasField("t"))]
    numbers = [numpy_helper.to_array(tensor).astype(float).ravel() for tensor in tensors]
    numbers += [np.asarray(attribute.floats, dtype=float) for node in graph.node for attribute in node.attribute]
    numbers += [np.asarray([attribute.f]) for node in graph.node for attribute in node.attribute if attribute.HasField("f")]
    stored = np.concatenate(numbers)
    expected = np.array([*model.coefficients, model.intercept])
    assert stored.size == expected.size
    np.testing.assert_allclose(np.sort(stored), np.sort(expected), rtol=1e-5)
    assert not _dataset_values() & {round(float(value), 6) for value in stored}


# ----------------------------------------------------------------------------
# Destinations and errors
# ----------------------------------------------------------------------------


def test_folder_with_spaces_accents_and_enye(tmp_path: Path, model: RegressionModel) -> None:
    """Paths with spaces, accents and ñ work."""
    folder = tmp_path / "Mis modelos de energía" / "Año 2026 – diseño"
    folder.mkdir(parents=True)
    files = export_model(model, folder / "predicción del consumo")
    assert files.model == folder / "predicción del consumo.onnx"
    assert files.model.is_file() and files.metadata.is_file()
    assert _load(files.metadata)["model_file"] == "predicción del consumo.onnx"


def test_existing_files_are_replaced(tmp_path: Path, model: RegressionModel) -> None:
    """Saving again over the same files replaces them; the GUI asks before doing so."""
    files = model_files(tmp_path / "modelo.onnx")
    files.model.write_bytes(b"old")
    files.metadata.write_text("{}", encoding="utf-8")
    export_model(model, files.model, "Nueva versión")
    assert files.model.read_bytes() != b"old"
    assert _load(files.metadata)["description"] == "Nueva versión"


def test_missing_folder_raises_a_clear_error(tmp_path: Path, model: RegressionModel) -> None:
    """A folder that does not exist is reported and nothing is written."""
    with pytest.raises(ModelExportError, match="does not exist"):
        export_model(model, tmp_path / "missing" / "modelo.onnx")
    assert not any(tmp_path.iterdir())


def test_permission_error_keeps_existing_files_and_leaves_no_temporary_files(
    tmp_path: Path, model: RegressionModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder that cannot be written is reported clearly, and the previous files are untouched."""
    files = model_files(tmp_path / "modelo.onnx")
    files.model.write_bytes(b"old")

    def denied(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(export_module.os, "replace", denied)
    with pytest.raises(ModelExportError, match="no permission"):
        export_model(model, files.model)
    assert files.model.read_bytes() == b"old"
    assert {path.name for path in tmp_path.iterdir()} == {"modelo.onnx"}


def test_build_metadata_keeps_non_text_column_labels(model: RegressionModel) -> None:
    """Column labels that JSON can represent, such as numbers, are kept as they are."""
    numeric = create_model(TRAIN.set_axis([0, 1, "consumo"], axis=1), TEST.set_axis([0, 1, "consumo"], axis=1), [0, 1], "consumo")
    metadata = build_metadata(numeric)
    assert metadata["input_columns"] == [0, 1]
    json.dumps(metadata, allow_nan=False)
