"""Unit tests for loading a regression model saved as an ONNX file, using models trained in each test."""

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest
from pandas import DataFrame

from sustentable.core.io import import_model as import_module
from sustentable.core.io.export_model import export_model
from sustentable.core.io.import_model import ModelImportError, SavedModel, import_model
from sustentable.core.model.regression import RegressionModel, create_model

TRAIN = DataFrame(
    {
        "temperatura": [11.125, 13.375, 17.625, 19.875, 23.125, 29.375],
        "humedad": [41.5, 47.25, 52.75, 49.5, 63.25, 58.75],
        "consumo": [101.0625, 117.4375, 139.8125, 142.1875, 171.5625, 189.9375],
    }
)
TEST = DataFrame({"temperatura": [15.625, 21.875], "humedad": [44.75, 61.25], "consumo": [126.3125, 163.6875]})
NEW_ROWS = DataFrame({"temperatura": [-5.0, 0.0, 18.5, 45.25], "humedad": [0.0, 12.5, 50.0, 99.75]})
# Marks a metadata entry that ``_rewrite`` removes from the file.
REMOVE = object()


@pytest.fixture
def model() -> RegressionModel:
    """A multiple regression trained on the small dataset above."""
    return create_model(TRAIN, TEST, ["temperatura", "humedad"], "consumo")


@pytest.fixture
def saved(tmp_path: Path, model: RegressionModel) -> Path:
    """The model above saved with a description."""
    return export_model(model, tmp_path / "modelo.onnx", "Predice el consumo.").model


def _metadata(path: Path) -> dict[str, object]:
    """Every metadata entry of the saved file, decoded."""
    return {prop.key: json.loads(prop.value) for prop in onnx.load(path).metadata_props}


def _rewrite(path: Path, **entries: object) -> Path:
    """Replace metadata entries of the saved file, written as JSON unless given as ``bytes``, or remove them."""
    onnx_model = onnx.load(path)
    props = {prop.key: prop.value for prop in onnx_model.metadata_props}
    for key, value in entries.items():
        if value is REMOVE:
            props.pop(key, None)
        else:
            props[key] = value.decode() if isinstance(value, bytes) else json.dumps(value)
    del onnx_model.metadata_props[:]
    onnx.helper.set_model_props(onnx_model, props)
    onnx.save(onnx_model, path)
    return path


def _assert_same_model(loaded: RegressionModel, original: RegressionModel) -> None:
    """Columns, parameters, metrics and predictions are exactly the original ones."""
    assert (loaded.inputs, loaded.output) == (original.inputs, original.output)
    assert (loaded.coefficients, loaded.intercept) == (original.coefficients, original.intercept)
    assert (loaded.train_metrics, loaded.test_metrics) == (original.train_metrics, original.test_metrics)
    assert loaded.formula == original.formula
    np.testing.assert_array_equal(loaded.predict(NEW_ROWS[list(original.inputs)]), original.predict(NEW_ROWS[list(original.inputs)]))


# ----------------------------------------------------------------------------
# Loading what was saved
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("description", ["Predice el consumo según la temperatura y la humedad.", ""])
def test_saved_model_is_loaded_with_all_its_information(tmp_path: Path, model: RegressionModel, description: str) -> None:
    """Columns, coefficients, metrics and description, even an empty one, survive saving and loading."""
    path = export_model(model, tmp_path / "modelo.onnx", description).model

    loaded = import_model(path)

    assert isinstance(loaded, SavedModel)
    assert loaded.description == description
    assert loaded.path == path.resolve()
    _assert_same_model(loaded.model, model)


def test_simple_model_with_an_undefined_metric_is_loaded(tmp_path: Path) -> None:
    """R² over a single test row is saved as null and loaded back as undefined."""
    model = create_model(TRAIN, TEST.head(1), ["temperatura"], "consumo")
    loaded = import_model(export_model(model, tmp_path / "simple.onnx").model).model

    assert loaded.is_simple and math.isnan(loaded.test_metrics.r2)
    assert loaded.test_metrics.mse == model.test_metrics.mse and loaded.test_metrics.rows == 1
    assert loaded.train_metrics == model.train_metrics
    np.testing.assert_array_equal(loaded.predict(NEW_ROWS), model.predict(NEW_ROWS))


def test_numeric_column_labels_are_loaded_as_they_were_saved(tmp_path: Path) -> None:
    """Labels that are numbers stay numbers, so the model still selects those columns of a frame."""
    rename = {"temperatura": 0, "humedad": 1}
    model = create_model(TRAIN.rename(columns=rename), TEST.rename(columns=rename), [0, 1], "consumo")
    loaded = import_model(export_model(model, tmp_path / "numeric.onnx").model).model

    assert loaded.inputs == (0, 1)
    np.testing.assert_array_equal(loaded.predict(NEW_ROWS.rename(columns=rename)), model.predict(NEW_ROWS.rename(columns=rename)))


def test_model_saved_again_with_a_new_description_keeps_everything_else(saved: Path, model: RegressionModel) -> None:
    """Only the description changes; the coefficients, columns, metrics and the rest of the metadata stay the same."""
    before = _metadata(saved)
    loaded = import_model(saved)

    export_model(loaded.model, saved, "Nueva descripción del modelo.")
    reloaded = import_model(saved)

    assert reloaded.description == "Nueva descripción del modelo."
    _assert_same_model(reloaded.model, model)
    after = _metadata(saved)
    assert after["description"] == "Nueva descripción del modelo."
    assert {key: value for key, value in after.items() if key != "description"} == {
        key: value for key, value in before.items() if key != "description"
    }


def test_folder_with_spaces_accents_and_enye(tmp_path: Path, model: RegressionModel) -> None:
    """Models are loaded from any folder, including names with spaces, accents and ñ."""
    folder = tmp_path / "Mis modelos de energía" / "Año 2026 – diseño"
    folder.mkdir(parents=True)
    path = export_model(model, folder / "predicción del consumo.onnx", "Descripción con ñ.").model

    loaded = import_model(path)

    assert loaded.path == path.resolve() and loaded.description == "Descripción con ñ."
    _assert_same_model(loaded.model, model)


def test_relative_path_is_loaded_and_reported_as_absolute(tmp_path: Path, model: RegressionModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """A path relative to the working directory is accepted, and the loaded model remembers the absolute one."""
    export_model(model, tmp_path / "modelo.onnx")
    monkeypatch.chdir(tmp_path)

    loaded = import_model("modelo.onnx")

    assert loaded.path.is_absolute() and loaded.path == (tmp_path / "modelo.onnx").resolve()


# ----------------------------------------------------------------------------
# Files that cannot be loaded
# ----------------------------------------------------------------------------


def _load_error(path: Path) -> str:
    """Load ``path``, which must fail with a controlled error that names the file; return its message."""
    with pytest.raises(ModelImportError) as caught:
        import_model(path)
    message = str(caught.value)
    assert str(path.resolve()) in message
    return message


def _missing(tmp_path: Path, saved: Path) -> Path:
    """A path where there is nothing."""
    return tmp_path / "missing.onnx"


def _folder(tmp_path: Path, saved: Path) -> Path:
    """A folder named like a model."""
    folder = tmp_path / "folder.onnx"
    folder.mkdir()
    return folder


def _empty(tmp_path: Path, saved: Path) -> Path:
    """A model file of zero bytes."""
    saved.write_bytes(b"")
    return saved


def _text(tmp_path: Path, saved: Path) -> Path:
    """A CSV dataset with the model extension."""
    saved.write_text("temperatura,humedad,consumo\n11.1,41.5,101.0\n", encoding="utf-8")
    return saved


def _binary(tmp_path: Path, saved: Path) -> Path:
    """The start of a PNG image with the model extension."""
    saved.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe" * 20)
    return saved


def _truncated(tmp_path: Path, saved: Path) -> Path:
    """A model cut in half, as an interrupted copy leaves it."""
    saved.write_bytes(saved.read_bytes()[: saved.stat().st_size // 2])
    return saved


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (_missing, "does not exist"),
        (_folder, "is not a file"),
        (_empty, "is empty"),
        (_text, "is not a valid ONNX model"),
        (_binary, "is not a valid ONNX model"),
        (_truncated, ""),
    ],
    ids=["missing", "folder", "empty", "text", "binary", "truncated"],
)
def test_unreadable_or_damaged_files_are_reported(tmp_path: Path, saved: Path, damage: Callable[[Path, Path], Path], message: str) -> None:
    """Files that are not ONNX models give a controlled error instead of an exception from the libraries."""
    assert message in _load_error(damage(tmp_path, saved))


def test_onnx_model_not_saved_by_the_application_is_reported(tmp_path: Path, saved: Path) -> None:
    """A valid ONNX model without this application's metadata has nothing to show."""
    onnx_model = onnx.load(saved)
    del onnx_model.metadata_props[:]
    onnx.save(onnx_model, saved)
    assert "not saved by Sustentable" in _load_error(saved)
    assert "not saved by Sustentable" in _load_error(_rewrite(saved, format="ONNX", model_type="RandomForest"))


def test_newer_format_version_is_reported(saved: Path) -> None:
    """A file written with an incompatible version of the format is not misread."""
    assert "version 2.0" in _load_error(_rewrite(saved, version="2.0"))
    assert import_model(_rewrite(saved, version="1.7")).description == "Predice el consumo."


@pytest.mark.parametrize(
    ("entries", "what"),
    [
        ({"input_columns": REMOVE}, "input columns"),
        ({"output_columns": REMOVE}, "output column"),
        ({"parameters": REMOVE}, "coefficients"),
        ({"parameters": {"coefficients": [1.0, 2.0]}}, "intercept"),
        ({"metrics": REMOVE}, "training R²"),
        ({"metrics": {"train": {"r2": 1.0, "mse": 0.0}}}, "test R²"),
        ({"training": REMOVE}, "number of training rows"),
        ({"description": REMOVE}, "description"),
        ({"version": REMOVE}, "format version"),
    ],
)
def test_missing_information_is_reported(saved: Path, entries: dict[str, object], what: str) -> None:
    """Each piece of information the interface needs must be in the file."""
    assert f"incomplete: it does not contain the {what}" in _load_error(_rewrite(saved, **entries))


@pytest.mark.parametrize(
    ("entries", "what"),
    [
        ({"input_columns": []}, "input columns"),
        ({"input_columns": "temperatura"}, "input columns"),
        ({"input_columns": [["temperatura"], "humedad"]}, "input columns"),
        ({"output_columns": ["consumo", "otro"]}, "output column"),
        ({"parameters": {"coefficients": [1.0], "intercept": 2.0}}, "coefficients"),
        ({"parameters": {"coefficients": [1.0, "2"], "intercept": 2.0}}, "coefficients"),
        ({"parameters": {"coefficients": [1.0, True], "intercept": 2.0}}, "coefficients"),
        ({"parameters": {"coefficients": [1.0, 2.0], "intercept": None}}, "intercept"),
        ({"parameters": b'{"coefficients": [1.0, NaN], "intercept": 2.0}'}, "coefficients"),
        ({"parameters": b'{"coefficients": [1.0, 2.0], "intercept": 1' + b"0" * 400 + b"}"}, "intercept"),
        ({"metrics": {"train": {"r2": "high", "mse": 0.0}, "test": {"r2": 1.0, "mse": 0.0}}}, "training R²"),
        ({"training": {"train_rows": 0, "test_rows": 2}}, "number of training rows"),
        ({"training": {"train_rows": 6, "test_rows": 2.5}}, "number of test rows"),
        ({"description": 5}, "description"),
        ({"description": b"{not json"}, "description"),
        ({"input_columns": ["temperatura", "temperatura"]}, "input and output columns"),
        ({"output_columns": ["humedad"]}, "input and output columns"),
    ],
)
def test_invalid_information_is_reported(saved: Path, entries: dict[str, object], what: str) -> None:
    """Values of the wrong type or that no trained model could have are refused, not shown."""
    assert f"damaged: it has an invalid value for the {what}" in _load_error(_rewrite(saved, **entries))


def test_metadata_that_does_not_match_the_graph_is_reported(saved: Path, tmp_path: Path) -> None:
    """Changed coefficients, or the metadata of a model with other inputs, do not describe the model in the file."""
    metadata = _metadata(saved)
    parameters = metadata["parameters"]
    doubled = {"coefficients": [2 * value for value in parameters["coefficients"]], "intercept": parameters["intercept"]}
    assert "does not match" in _load_error(_rewrite(saved, parameters=doubled))

    simple = export_model(create_model(TRAIN, TEST, ["temperatura"], "consumo"), tmp_path / "simple.onnx").model
    assert "does not match" in _load_error(_rewrite(simple, **metadata))


def test_large_and_unreadable_files_are_reported(saved: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file too large to be a model is not read whole, and a file without read permission is reported."""
    monkeypatch.setattr(import_module, "MAX_MODEL_BYTES", saved.stat().st_size - 1)
    assert "too large" in _load_error(saved)
    monkeypatch.undo()

    def denied(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "open", denied)
    assert "permission denied" in _load_error(saved)
