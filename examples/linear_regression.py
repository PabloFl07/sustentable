"""Proof of concept: train, interpret and evaluate a linear regression model with scikit-learn."""

from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sustentable.core.readers import get_reader

DATA_FILE = "data/housing(1).csv"
FEATURE_COLUMN = "median_income"
TARGET_COLUMN = "median_house_value"
NEW_VALUE = 5.0  # New median_income value to predict
TEST_RATIO = 0.2


def data_preparation(data: DataFrame) -> tuple[DataFrame, DataFrame, Series, Series]:
    """Split the input and output columns into training and test sets."""
    X = data[[FEATURE_COLUMN]]
    y = data[TARGET_COLUMN]

    return _data_split(X, y)


def _data_split(X, y):

    return train_test_split(X, y, test_size=TEST_RATIO, random_state=42)


def train_model(X_train: DataFrame, y_train: Series) -> LinearRegression:
    """Train a linear regression model on the training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def model_formula(model: LinearRegression) -> str:
    """Return the model formula using the column names, the coefficient and the intercept."""
    return f"\n{TARGET_COLUMN} = {model.coef_[0]:.4f} * {FEATURE_COLUMN} + {model.intercept_:.4f}"


def evaluate_model(model: LinearRegression, X: DataFrame, y: Series, label: str) -> None:
    """Print the mean squared error and R^2 score of the model on the given data."""
    predictions = model.predict(X)

    print(f"\n[{label}] Mean Squared Error: {mean_squared_error(y, predictions):.4f}")
    print(f"[{label}] R^2 Score: {r2_score(y, predictions):.4f}")


def main() -> None:
    """Load the data, train the model, show its formula, evaluate it and make a new prediction."""
    data = get_reader(DATA_FILE).read()

    X_train, X_test, y_train, y_test = data_preparation(data)

    model = train_model(X_train, y_train)

    print(model_formula(model))

    evaluate_model(model, X_train, y_train, "train")
    evaluate_model(model, X_test, y_test, "test")

    new_input = DataFrame({FEATURE_COLUMN: [NEW_VALUE]})
    print(f"\nPrediction for {FEATURE_COLUMN} = {NEW_VALUE}: {model.predict(new_input)[0]:.4f}")


if __name__ == "__main__":
    main()
