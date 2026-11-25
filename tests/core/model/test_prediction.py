"""Unit tests for predicting the output of a model from the values typed by the user."""

import math

import numpy as np
import pytest
from pandas import DataFrame

from sustentable.core.model.prediction import PredictionInputError, format_prediction, parse_number, predict_value
from sustentable.core.model.regression import create_model


def _plane(a: list[float], b: list[float]) -> DataFrame:
    """Rows exactly on y = 3a - 2b + 5."""
    return DataFrame({"a": a, "b": b, "y": [3 * i - 2 * j + 5 for i, j in zip(a, b, strict=True)]})


@pytest.fixture
def plane_model():
    """Multiple regression trained on rows of y = 3a - 2b + 5, so its formula can be checked by hand."""
    return create_model(_plane([0, 1, 2, 3, 4, 5], [1, 0, 2, 5, 3, 4]), _plane([6, 7], [1, 8]), ["a", "b"], "y")


@pytest.fixture
def line_model():
    """Simple regression trained on rows of y = 2x + 1."""
    train = DataFrame({"x": [0, 1, 2, 3], "y": [1, 3, 5, 7]})
    test = DataFrame({"x": [4, 5], "y": [9, 11]})
    return create_model(train, test, ["x"], "y")


# ----------------------------------------------------------------------------
# Parsing numbers
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"), [("3.5", 3.5), ("3,5", 3.5), (" -2 ", -2.0), ("0", 0.0), ("-0,25", -0.25), (".5", 0.5), ("1e3", 1000.0)]
)
def test_numbers_accept_a_point_or_a_comma_as_decimal_separator(text: str, expected: float) -> None:
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "abc", "3,5,1", "1,234.5", "1.234,5", "1_000", "nan", "inf", "-Infinity", "3.5.1", "12a"])
def test_text_that_is_not_a_finite_number_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="not a valid number"):
        parse_number(text)


# ----------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------


def test_simple_prediction_matches_the_formula_by_hand(line_model) -> None:
    coefficient, intercept = line_model.coefficients[0], line_model.intercept
    assert predict_value(line_model, {"x": "10"}) == pytest.approx(coefficient * 10 + intercept)
    assert predict_value(line_model, {"x": "10"}) == pytest.approx(21.0)


def test_multiple_prediction_matches_the_formula_by_hand(plane_model) -> None:
    (ca, cb), intercept = plane_model.coefficients, plane_model.intercept
    expected = ca * 2.5 + cb * -1 + intercept
    assert predict_value(plane_model, {"a": "2,5", "b": "-1"}) == pytest.approx(expected)
    assert predict_value(plane_model, {"a": "2,5", "b": "-1"}) == pytest.approx(3 * 2.5 - 2 * -1 + 5)


def test_comma_and_point_give_the_same_prediction_and_numbers_are_accepted(plane_model) -> None:
    by_point = predict_value(plane_model, {"a": "1.5", "b": "0.25"})
    assert predict_value(plane_model, {"a": "1,5", "b": "0,25"}) == by_point
    assert predict_value(plane_model, {"a": 1.5, "b": 0.25}) == by_point


def test_prediction_matches_the_model_predict_and_ignores_extra_values(plane_model) -> None:
    value = predict_value(plane_model, {"b": "4", "a": "7", "y": "ignored", "other": 1})
    assert value == pytest.approx(float(plane_model.predict(DataFrame({"a": [7.0], "b": [4.0]}))[0]))
    assert isinstance(value, float)


def test_missing_values_are_named(plane_model) -> None:
    with pytest.raises(PredictionInputError) as caught:
        predict_value(plane_model, {"a": "  "})
    assert caught.value.missing == ("a", "b") and caught.value.invalid == ()
    assert str(caught.value) == "Enter a value for 'a' and 'b'."


def test_values_that_are_not_numbers_are_named(plane_model) -> None:
    with pytest.raises(PredictionInputError) as caught:
        predict_value(plane_model, {"a": "abc", "b": "2"})
    assert caught.value.invalid == ("a",) and caught.value.missing == ()
    assert str(caught.value) == "The value of 'a' is not a valid number."


def test_missing_and_invalid_values_are_reported_together(plane_model) -> None:
    with pytest.raises(PredictionInputError) as caught:
        predict_value(plane_model, {"a": math.nan, "b": ""})
    assert (caught.value.missing, caught.value.invalid) == (("b",), ("a",))
    assert str(caught.value) == "Enter a value for 'b'. The value of 'a' is not a valid number."


def test_error_is_a_value_error_and_a_retry_with_valid_values_works(plane_model) -> None:
    with pytest.raises(ValueError):
        predict_value(plane_model, {"a": "x", "b": "y"})
    assert predict_value(plane_model, {"a": "0", "b": "0"}) == pytest.approx(5.0)


def test_any_object_with_inputs_output_and_predict_can_be_used() -> None:
    """A model loaded from a file only needs the same interface as a created model."""

    class Loaded:
        inputs = ("a", "b")
        output = "y"

        def predict(self, frame: DataFrame) -> np.ndarray:
            return (frame["a"] * 10 + frame["b"]).to_numpy(dtype=float)

    assert predict_value(Loaded(), {"a": "1,5", "b": "2"}) == 17.0


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------


def test_result_uses_the_output_name_and_the_formula_decimals(line_model) -> None:
    assert format_prediction("y", 21.0) == "y = 21.0000"
    assert format_prediction("energy", -1.234567) == "energy = -1.2346"
    assert format_prediction("y", -0.00001) == "y = 0.0000"
    assert format_prediction("y", 1.23456, decimals=2) == "y = 1.23"
    value = predict_value(line_model, {"x": "10"})
    assert format_prediction(line_model.output, value) == "y = 21.0000"
