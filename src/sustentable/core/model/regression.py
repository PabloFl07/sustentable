"""Train a linear regression on the training subset, evaluate it on both subsets and describe it.

The same functions handle simple regression (one input column) and multiple regression (several input columns).
Nothing here depends on the GUI: the functions take the subsets and the selected columns, and return values the
interface only has to show. The subsets passed in are never modified.
"""

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Decimals shown for the coefficients, the intercept and the metrics.
DECIMALS = 4
# Shown instead of a metric that is not defined, such as R² over a single row.
UNDEFINED = "n/a"

type Point = tuple[float, float]


@dataclass(frozen=True)
class Metrics:
    """Error metrics of a model on one subset of ``rows`` rows.

    ``r2`` is the coefficient of determination and ``mse`` the mean squared error.
    ``r2`` is ``nan`` when the subset has a single row, because it is not defined there.
    """

    r2: float
    mse: float
    rows: int


@dataclass(frozen=True)
class RegressionModel:
    """Linear regression trained on the training subset, with its metrics on the training and test subsets.

    ``inputs`` and ``output`` are the labels of the columns it was trained with. ``predict`` takes new rows with the
    same input columns, so the model can be kept to make predictions or to be saved once it is created.
    """

    inputs: tuple[Hashable, ...]
    output: Hashable
    train_metrics: Metrics
    test_metrics: Metrics
    estimator: LinearRegression = field(repr=False, compare=False)

    @property
    def is_simple(self) -> bool:
        """Whether the model has a single input column."""
        return len(self.inputs) == 1

    @property
    def coefficients(self) -> tuple[float, ...]:
        """Coefficient of each input column, in the order of ``inputs``."""
        return tuple(float(value) for value in self.estimator.coef_)

    @property
    def intercept(self) -> float:
        """Independent term: the predicted output when every input is zero."""
        return float(self.estimator.intercept_)

    @property
    def formula(self) -> str:
        """Formula of the model with the column names and rounded values."""
        return format_formula(self.output, self.inputs, self.coefficients, self.intercept)

    def predict(self, frame: DataFrame) -> NDArray[np.float64]:
        """Predict the output for each row of ``frame``, which must contain the input columns.

        Raises:
            ValueError: If an input column is missing or repeated, is not numeric, or has missing or infinite values.
        """
        _check_columns(frame, self.inputs, "data")
        return self.estimator.predict(_values(frame, self.inputs, "data"))


@dataclass(frozen=True)
class FitPlot:
    """Points and line of the fit chart, in data units, and the title of each axis.

    In a simple regression the x axis is the input column, the y axis is the output column and the line is the
    regression line. In a multiple regression the x axis is the actual output, the y axis is the predicted output and
    the line marks where both are equal: the closer the points are to it, the better the model predicts.
    """

    x_label: str
    y_label: str
    line_label: str
    train_points: tuple[Point, ...]
    test_points: tuple[Point, ...]
    line: tuple[Point, Point]


def create_model(train: DataFrame, test: DataFrame, inputs: Sequence[Hashable], output: Hashable) -> RegressionModel:
    """Fit a linear regression with the training subset only, then evaluate it on both subsets.

    Args:
        train: Training subset. It is the only data used to fit the model.
        test: Test subset. It is only used to measure how well the model generalizes.
        inputs: Input columns, in the order the formula lists them. A single column gives a simple regression.
        output: Output column.

    Returns:
        The trained model, with its R² and mean squared error on each subset.

    Raises:
        ValueError: If no input is chosen, an input is repeated or is also the output, a column is missing or repeated
            in a subset, a column is not numeric or has missing or infinite values, the training subset has fewer
            than two rows, or the test subset is empty. The message says which one.
    """
    inputs = tuple(inputs)
    _check_selection(inputs, output)
    _check_columns(train, (*inputs, output), "training subset")
    _check_columns(test, (*inputs, output), "test subset")
    if len(train) < 2:
        raise ValueError("At least two training rows are needed to fit the model.")
    if test.empty:
        raise ValueError("The test subset is empty, so the model cannot be evaluated.")
    x_train, y_train = _inputs_and_output(train, inputs, output, "training subset")
    x_test, y_test = _inputs_and_output(test, inputs, output, "test subset")

    estimator = LinearRegression()
    try:
        estimator.fit(x_train, y_train)
    except ValueError as error:
        raise ValueError(f"The model could not be fitted: {error}") from error
    train_metrics = compute_metrics(y_train, estimator.predict(x_train))
    test_metrics = compute_metrics(y_test, estimator.predict(x_test))
    return RegressionModel(inputs, output, train_metrics, test_metrics, estimator)


def compute_metrics(actual: ArrayLike, predicted: ArrayLike) -> Metrics:
    """Compare predicted output values with the actual ones.

    Returns:
        The coefficient of determination (R²), the mean squared error and the number of values compared.
        R² is ``nan`` for a single value, because it is not defined there.

    Raises:
        ValueError: If there are no values or the two sequences have different lengths.
    """
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.size == 0:
        raise ValueError("At least one value is needed to compute the metrics.")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("There must be one predicted value per actual value.")
    r2 = float(r2_score(actual_values, predicted_values)) if actual_values.size >= 2 else math.nan
    return Metrics(r2=r2, mse=float(mean_squared_error(actual_values, predicted_values)), rows=int(actual_values.size))


def format_formula(
    output: Hashable, inputs: Sequence[Hashable], coefficients: Sequence[float], intercept: float, decimals: int = DECIMALS
) -> str:
    """Write ``<output> = <c1> * <input1> + <c2> * <input2> + ... + <intercept>`` with the column names.

    Every value is rounded to ``decimals`` decimals. Negative values are subtracted (``y = 2.0000 * x - 1.5000``)
    instead of added as ``+ -1.5000``; only a negative first coefficient keeps its sign in front of it.
    A value that rounds to zero is written without a sign.

    Raises:
        ValueError: If there is not exactly one coefficient per input column.
    """
    if len(coefficients) != len(inputs):
        raise ValueError("There must be exactly one coefficient per input column.")
    terms = [(coefficient, f" * {column}") for coefficient, column in zip(coefficients, inputs, strict=True)]
    terms.append((intercept, ""))
    text = ""
    for position, (value, name) in enumerate(terms):
        magnitude = f"{abs(value):.{decimals}f}"
        negative = value < 0 and float(magnitude) != 0
        if position == 0:
            sign = "-" if negative else ""
        else:
            sign = " - " if negative else " + "
        text += f"{sign}{magnitude}{name}"
    return f"{output} = {text}"


def format_metric(value: float, decimals: int = DECIMALS) -> str:
    """Round a metric for display, with thousands separators, or return ``UNDEFINED`` when it is ``nan``."""
    if math.isnan(value):
        return UNDEFINED
    text = f"{value:,.{decimals}f}"
    # A tiny negative value would otherwise be shown as "-0.0000".
    return text.removeprefix("-") if float(text.replace(",", "")) == 0 else text


def fit_plot(model: RegressionModel, train: DataFrame, test: DataFrame) -> FitPlot:
    """Prepare the chart that compares ``model`` with the training and test subsets.

    Raises:
        ValueError: If a subset lacks a column of the model or has values the model cannot use.
    """
    columns = (*model.inputs, model.output)
    _check_columns(train, columns, "training subset")
    _check_columns(test, columns, "test subset")
    x_train, y_train = _inputs_and_output(train, model.inputs, model.output, "training subset")
    x_test, y_test = _inputs_and_output(test, model.inputs, model.output, "test subset")

    if model.is_simple:
        # The regression line spans the input values of both subsets.
        low = float(min(x_train.min(), x_test.min()))
        high = float(max(x_train.max(), x_test.max()))
        start, end = (float(value) for value in model.estimator.predict(np.array([[low], [high]])))
        return FitPlot(
            x_label=str(model.inputs[0]),
            y_label=str(model.output),
            line_label="Regression line",
            train_points=_points(x_train[:, 0], y_train),
            test_points=_points(x_test[:, 0], y_test),
            line=((low, start), (high, end)),
        )

    predicted_train, predicted_test = model.estimator.predict(x_train), model.estimator.predict(x_test)
    values = np.concatenate([y_train, y_test, predicted_train, predicted_test])
    low, high = float(values.min()), float(values.max())
    return FitPlot(
        x_label=f"Actual value ({model.output})",
        y_label=f"Predicted value ({model.output})",
        line_label="Predicted = actual",
        train_points=_points(y_train, predicted_train),
        test_points=_points(y_test, predicted_test),
        line=((low, low), (high, high)),
    )


def _check_selection(inputs: tuple[Hashable, ...], output: Hashable) -> None:
    """Reject an empty or repeated input selection, and an output that is also an input."""
    if not inputs:
        raise ValueError("Choose at least one input column.")
    if len(set(inputs)) < len(inputs):
        raise ValueError("Each input column can only be chosen once.")
    if output in inputs:
        raise ValueError(f"Column '{output}' cannot be both an input and the output.")


def _check_columns(frame: DataFrame, columns: Sequence[Hashable], where: str) -> None:
    """Check that every column appears exactly once in ``frame``, so it selects a single series."""
    labels = list(frame.columns)
    for column in columns:
        count = labels.count(column)
        if not count:
            raise ValueError(f"Column '{column}' is not available in the {where}.")
        if count > 1:
            raise ValueError(f"Column '{column}' appears more than once in the {where}. Rename the repeated columns to use it.")


def _values(frame: DataFrame, columns: Sequence[Hashable], where: str) -> NDArray[np.float64]:
    """Return ``columns`` as a float matrix, rejecting text, missing and infinite values with an explanatory message."""
    for column in columns:
        series = frame[column]
        if not is_numeric_dtype(series):
            raise ValueError(f"Column '{column}' is not numeric. A linear regression can only use numeric columns.")
        if missing := int(series.isna().sum()):
            noun = "value" if missing == 1 else "values"
            raise ValueError(f"Column '{column}' has {missing:,} missing {noun} in the {where}, and a linear regression cannot use them.")
    values = frame[list(columns)].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=0)
    if not finite.all():
        column = columns[int(np.argmin(finite))]
        raise ValueError(f"Column '{column}' has infinite values in the {where}, so the model cannot use it.")
    return values


def _inputs_and_output(
    frame: DataFrame, inputs: Sequence[Hashable], output: Hashable, where: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the input matrix and the output vector of ``frame``."""
    values = _values(frame, (*inputs, output), where)
    return values[:, :-1], values[:, -1]


def _points(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[Point, ...]:
    """Pair the x and y values as plain Python floats."""
    return tuple(zip(x.tolist(), y.tolist(), strict=True))
