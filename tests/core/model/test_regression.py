"""Unit tests for training, evaluating and describing linear regression models."""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from sustentable.core.model.regression import UNDEFINED, compute_metrics, create_model, fit_plot, format_formula, format_metric

NAN = math.nan


def _line(x: list[float]) -> DataFrame:
    """Rows that lie exactly on the line y = 2x + 1."""
    return DataFrame({"x": x, "y": [2 * value + 1 for value in x]})


def _plane(a: list[float], b: list[float], offset: float = 0.0) -> DataFrame:
    """Rows on y = 3a - 2b + 5, moved up by ``offset``, plus an unselected text column."""
    return DataFrame({"a": a, "b": b, "y": [3 * i - 2 * j + 5 + offset for i, j in zip(a, b, strict=True)], "note": "row"})


@pytest.fixture
def plane_train() -> DataFrame:
    """Training rows whose inputs are not collinear, so both coefficients can be recovered."""
    return _plane([0, 1, 2, 3, 4, 5], [1, 0, 2, 5, 3, 4])


@pytest.fixture
def plane_test() -> DataFrame:
    """Test rows three units above the plane, so the model misses each of them by 3."""
    return _plane([6, 7, 8], [1, 8, 2], offset=3)


# ----------------------------------------------------------------------------
# Training and metrics
# ----------------------------------------------------------------------------


def test_simple_regression_recovers_the_line() -> None:
    """One input column gives a simple regression with the exact slope and intercept."""
    model = create_model(_line([0, 1, 2, 3]), _line([4, 5]), ["x"], "y")

    assert model.is_simple
    assert (model.inputs, model.output) == (("x",), "y")
    assert model.coefficients == pytest.approx((2.0,))
    assert model.intercept == pytest.approx(1.0)
    assert model.formula == "y = 2.0000 * x + 1.0000"
    for metrics, rows in ((model.train_metrics, 4), (model.test_metrics, 2)):
        assert metrics.r2 == pytest.approx(1.0)
        assert metrics.mse == pytest.approx(0.0, abs=1e-12)
        assert metrics.rows == rows


def test_model_is_fitted_with_the_training_subset_only() -> None:
    """Test rows with a different relation change the test metrics, never the coefficients."""
    train = _line([0, 1, 2, 3])
    test = DataFrame({"x": [4.0, 5.0], "y": [0.0, -10.0]})

    model = create_model(train, test, ["x"], "y")

    assert model.coefficients == pytest.approx((2.0,))
    assert model.intercept == pytest.approx(1.0)
    assert model.train_metrics.r2 == pytest.approx(1.0)
    # Predictions 9 and 11 miss by 9 and 21; the test values have a mean of -5.
    assert model.test_metrics.mse == pytest.approx((9**2 + 21**2) / 2)
    assert model.test_metrics.r2 == pytest.approx(1 - (9**2 + 21**2) / (5**2 + 5**2))


def test_multiple_regression_subtracts_the_negative_coefficient(plane_train: DataFrame, plane_test: DataFrame) -> None:
    """Several inputs give a multiple regression; its negative coefficient is written as a subtraction."""
    model = create_model(plane_train, plane_test, ["a", "b"], "y")

    assert not model.is_simple
    assert model.coefficients == pytest.approx((3.0, -2.0))
    assert model.intercept == pytest.approx(5.0)
    assert model.formula == "y = 3.0000 * a - 2.0000 * b + 5.0000"
    assert "+ -" not in model.formula
    assert model.train_metrics.mse == pytest.approx(0.0, abs=1e-12)
    assert model.test_metrics.mse == pytest.approx(9.0)


@pytest.mark.parametrize(
    ("slope", "intercept", "formula"),
    [(0.5, -4.0, "y = 0.5000 * x - 4.0000"), (-1.5, 2.0, "y = -1.5000 * x + 2.0000"), (-0.25, -7.0, "y = -0.2500 * x - 7.0000")],
    ids=["negative-intercept", "negative-first-coefficient", "both-negative"],
)
def test_trained_formula_with_negative_values(slope: float, intercept: float, formula: str) -> None:
    """Negative values found by training are shown with a minus sign, never as ``+ -``."""

    def rows(x: list[float]) -> DataFrame:
        return DataFrame({"x": x, "y": [slope * value + intercept for value in x]})

    model = create_model(rows([0, 2, 4, 6]), rows([1, 3]), ["x"], "y")

    assert model.formula == formula


def test_metrics_of_known_predictions() -> None:
    """R² and the mean squared error match a hand calculation."""
    metrics = compute_metrics([1, 2, 3, 4], [1, 2, 3, 5])

    # One error of 1 over four values; the actual values vary by 5 around their mean of 2.5.
    assert metrics.mse == pytest.approx(0.25)
    assert metrics.r2 == pytest.approx(1 - 1 / 5)
    assert metrics.rows == 4


def test_r2_is_undefined_for_a_single_row() -> None:
    """A single test row still has an error, but R² is shown as not defined, without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = create_model(_line([0, 1, 2]), DataFrame({"x": [3.0], "y": [9.0]}), ["x"], "y")

    assert math.isnan(model.test_metrics.r2)
    assert model.test_metrics.mse == pytest.approx(4.0)
    assert format_metric(model.test_metrics.r2) == UNDEFINED


@pytest.mark.parametrize(("actual", "predicted"), [([], []), ([1, 2], [1])])
def test_metrics_need_matching_values(actual: list[float], predicted: list[float]) -> None:
    """No values, or a different number of predicted and actual values, is an error."""
    with pytest.raises(ValueError):
        compute_metrics(actual, predicted)


def test_nullable_integer_and_boolean_columns_can_be_used() -> None:
    """Numeric columns of any type are accepted when they have no missing values."""
    train = DataFrame({"n": pd.array([0, 1, 2, 3], dtype="Int64"), "flag": [False, True, False, True], "y": [1.0, 4.0, 5.0, 8.0]})
    test = DataFrame({"n": pd.array([4], dtype="Int64"), "flag": [False], "y": [9.0]})

    model = create_model(train, test, ["n", "flag"], "y")

    # y = 2n + 1 + flag
    assert model.coefficients == pytest.approx((2.0, 1.0))
    assert model.intercept == pytest.approx(1.0)


def test_inputs_are_not_modified(plane_train: DataFrame, plane_test: DataFrame) -> None:
    """Training, evaluating and plotting work on copies of the subsets."""
    before = plane_train.copy(), plane_test.copy()

    model = create_model(plane_train, plane_test, ["a", "b"], "y")
    fit_plot(model, plane_train, plane_test)

    pd.testing.assert_frame_equal(plane_train, before[0])
    pd.testing.assert_frame_equal(plane_test, before[1])


# ----------------------------------------------------------------------------
# Invalid selections and data
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "output", "message"),
    [
        ([], "y", "at least one input"),
        (["a", "a"], "y", "only be chosen once"),
        (["a", "y"], "y", "both an input and the output"),
        (["missing"], "y", "'missing' is not available in the training subset"),
        (["a"], "note", "'note' is not numeric"),
    ],
)
def test_invalid_selection_is_rejected(plane_train: DataFrame, plane_test: DataFrame, inputs: list[str], output: str, message: str) -> None:
    """Each problem with the selected columns is explained."""
    with pytest.raises(ValueError, match=message):
        create_model(plane_train, plane_test, inputs, output)


@pytest.mark.parametrize(
    ("train", "test", "message"),
    [
        (DataFrame({"x": [0.0, NAN, 2.0], "y": [1.0, 3.0, 5.0]}), _line([3]), "1 missing value in the training subset"),
        (_line([0, 1, 2]), DataFrame({"x": [3.0, 4.0], "y": [NAN, NAN]}), "'y' has 2 missing values in the test subset"),
        (DataFrame({"x": [0.0, math.inf], "y": [1.0, 3.0]}), _line([3]), "'x' has infinite values in the training subset"),
        (_line([0, 1, 2]), DataFrame({"y": [7.0]}), "'x' is not available in the test subset"),
        (DataFrame([[0.0, 1.0, 1.0], [1.0, 2.0, 3.0]], columns=["x", "x", "y"]), _line([3]), "'x' appears more than once"),
        (_line([0]), _line([1]), "At least two training rows"),
        (_line([0, 1]), _line([]), "test subset is empty"),
    ],
    ids=["missing-train", "missing-test", "infinite", "absent-in-test", "repeated-label", "one-training-row", "empty-test"],
)
def test_unusable_data_is_rejected(train: DataFrame, test: DataFrame, message: str) -> None:
    """Data the model cannot be fitted or evaluated with raises an explanatory ``ValueError``."""
    with pytest.raises(ValueError, match=message):
        create_model(train, test, ["x"], "y")


# ----------------------------------------------------------------------------
# Formula and metric text
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "inputs", "coefficients", "intercept", "formula"),
    [
        ("y", ["x"], [2.0], 1.0, "y = 2.0000 * x + 1.0000"),
        ("y", ["x"], [-1.5], 2.0, "y = -1.5000 * x + 2.0000"),
        ("y", ["a", "b", "c"], [3.0, -2.0, -0.5], -5.0, "y = 3.0000 * a - 2.0000 * b - 0.5000 * c - 5.0000"),
        ("y", ["x"], [1.23456], 9.87654, "y = 1.2346 * x + 9.8765"),
        ("y", ["x"], [-0.00004], -0.00004, "y = 0.0000 * x + 0.0000"),
        ("energy use", ["floor area", 42], [0.5, -0.25], 10.0, "energy use = 0.5000 * floor area - 0.2500 * 42 + 10.0000"),
    ],
    ids=["positive", "negative-first", "negative-later-and-intercept", "rounded", "rounds-to-zero", "column-names"],
)
def test_formula_format(output: str, inputs: list[object], coefficients: list[float], intercept: float, formula: str) -> None:
    """Values are rounded to a fixed number of decimals and negative ones are subtracted."""
    assert format_formula(output, inputs, coefficients, intercept) == formula


def test_formula_decimals_can_be_changed() -> None:
    """The number of decimals is a parameter, four by default."""
    assert format_formula("y", ["x"], [2 / 3], -1 / 3, decimals=2) == "y = 0.67 * x - 0.33"


def test_formula_needs_one_coefficient_per_input() -> None:
    """A coefficient without its column, or the other way round, is an error."""
    with pytest.raises(ValueError, match="one coefficient per input"):
        format_formula("y", ["a", "b"], [1.0], 0.0)


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.98766, "0.9877"), (1234567.891, "1,234,567.8910"), (-0.5, "-0.5000"), (-0.00001, "0.0000"), (NAN, UNDEFINED)],
)
def test_metric_format(value: float, text: str) -> None:
    """Metrics are rounded, grouped by thousands, and ``nan`` is shown as not defined."""
    assert format_metric(value) == text


# ----------------------------------------------------------------------------
# Predictions and chart data
# ----------------------------------------------------------------------------


def test_predict_uses_the_input_columns_by_name(plane_train: DataFrame, plane_test: DataFrame) -> None:
    """New rows are predicted by column name, whatever their order and extra columns."""
    model = create_model(plane_train, plane_test, ["a", "b"], "y")

    predicted = model.predict(DataFrame({"other": ["x", "y"], "b": [0.0, 1.0], "a": [1.0, 0.0]}))

    assert predicted == pytest.approx([8.0, 3.0])
    with pytest.raises(ValueError, match="'b' is not available in the data"):
        model.predict(DataFrame({"a": [1.0]}))


def test_simple_fit_plot_shows_both_subsets_and_the_regression_line() -> None:
    """The input is on the x axis, the output on the y axis, and the line spans the inputs of both subsets."""
    train = _line([0, 1, 2, 3])
    test = DataFrame({"x": [-2.0, 6.0], "y": [0.0, 10.0]})
    model = create_model(train, test, ["x"], "y")

    plot = fit_plot(model, train, test)

    assert (plot.x_label, plot.y_label, plot.line_label) == ("x", "y", "Regression line")
    assert plot.train_points == ((0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0))
    assert plot.test_points == ((-2.0, 0.0), (6.0, 10.0))
    assert all(type(value) is float for point in (*plot.train_points, *plot.test_points, *plot.line) for value in point)
    start, end = plot.line
    assert start == pytest.approx((-2.0, -3.0))
    assert end == pytest.approx((6.0, 13.0))


def test_multiple_fit_plot_compares_actual_and_predicted_values(plane_train: DataFrame, plane_test: DataFrame) -> None:
    """Actual values are on the x axis, predictions on the y axis, and the line marks where both are equal."""
    model = create_model(plane_train, plane_test, ["a", "b"], "y")

    plot = fit_plot(model, plane_train, plane_test)

    assert (plot.x_label, plot.y_label, plot.line_label) == ("Actual value (y)", "Predicted value (y)", "Predicted = actual")
    for points, subset in ((plot.train_points, plane_train), (plot.test_points, plane_test)):
        actual, predicted = zip(*points, strict=True)
        assert list(actual) == subset.y.tolist()
        assert predicted == pytest.approx(model.predict(subset))
    values = np.concatenate([plane_train.y, plane_test.y, model.predict(plane_train), model.predict(plane_test)])
    (start_x, start_y), (end_x, end_y) = plot.line
    assert (start_x, start_y, end_x, end_y) == pytest.approx((values.min(), values.min(), values.max(), values.max()))
