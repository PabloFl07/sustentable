"""Proof of concept: train, interpret and evaluate simple and multiple linear regression models with scikit-learn.

A simple regression is a multiple regression with a single feature, so both cases share the same functions.
"""

from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sustentable.core.io.readers import get_reader

DATA_FILE = "data/housing(1).csv"
TARGET_COLUMN = "median_house_value"
TEST_RATIO = 0.2

# Each experiment maps its feature columns to the new values to predict.
EXPERIMENTS = [
    {"median_income": 5.0},
    {"median_income": 5.0, "housing_median_age": 30.0, "total_rooms": 2000.0},
]


def data_preparation(data: DataFrame, features: list[str]) -> tuple[DataFrame, DataFrame, Series, Series]:
    """Split the input and output columns into training and test sets."""
    X = data[features]
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
    """Return the model formula using the column names, the coefficients and the intercept."""
    terms = " + ".join(f"{coef:.4f} * {name}" for coef, name in zip(model.coef_, model.feature_names_in_, strict=True))

    return f"\n{TARGET_COLUMN} = {terms} + {model.intercept_:.4f}"


def evaluate_model(model: LinearRegression, X: DataFrame, y: Series, label: str) -> None:
    """Print the mean squared error and R^2 score of the model on the given data."""
    predictions = model.predict(X)

    print(f"\n[{label}] Mean Squared Error: {mean_squared_error(y, predictions):.4f}")
    print(f"[{label}] R^2 Score: {r2_score(y, predictions):.4f}")


def main() -> None:
    """Load the data and, for each experiment, train a model, show its formula, evaluate it and make a new prediction."""
    data = get_reader(DATA_FILE).read()

    for new_values in EXPERIMENTS:
        features = list(new_values)

        X_train, X_test, y_train, y_test = data_preparation(data, features)

        model = train_model(X_train, y_train)

        print(model_formula(model))

        evaluate_model(model, X_train, y_train, "train")
        evaluate_model(model, X_test, y_test, "test")

        new_input = DataFrame([new_values])
        print(f"\nPrediction for {new_values}: {model.predict(new_input)[0]:.4f}")


if __name__ == "__main__":
    main()
