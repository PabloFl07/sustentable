import pandas as pd
import numpy as np


def mean(dataframe: pd.DataFrame, column):
    """
    Replaces missing values in the selected numerical columns
    with their respective mean values.

    The mean is calculated using the provided DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data
            to be processed.
        column (list or str): Column or list of columns in which
            missing values should be replaced.

    Returns:
        pd.DataFrame: DataFrame containing the selected columns
            with missing values replaced by their mean.
    """
    values = dataframe[column].mean(numeric_only=True)

    numeric_columns = values.index.tolist()

    return dataframe[numeric_columns].fillna(values)


def delete_strategy(dataframe: pd.DataFrame, column):
    """
    Removes rows containing missing values in the selected columns.

    A row is removed if at least one of the specified columns
    contains a missing value.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data
            to be processed.
        column (list or str): Column or list of columns used
            to identify rows with missing values.

    Returns:
        pd.DataFrame: DataFrame with rows containing missing
            values in the selected columns removed.
    """
    return dataframe.dropna(subset=column)


def median(dataframe: pd.DataFrame, column):
    """
    Replaces missing values in the selected numerical columns
    with their respective median values.

    The median is calculated using the provided DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data
            to be processed.
        column (list or str): Column or list of columns in which
            missing values should be replaced.

    Returns:
        pd.DataFrame: DataFrame containing the selected columns
            with missing values replaced by their median.
    """
    values = dataframe[column].median(numeric_only=True)

    numeric_columns = values.index.tolist()

    return dataframe[numeric_columns].fillna(values)


def constant(dataframe: pd.DataFrame, column, constant_value):
    """
    Replaces missing values in the selected columns with
    a user-defined constant value.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data
            to be processed.
        column (list or str): Column or list of columns in which
            missing values should be replaced.
        constant_value (Any): Value used to replace missing values.

    Returns:
        pd.DataFrame: DataFrame containing the selected columns
            with missing values replaced by the specified constant.
    """
    return dataframe[column].fillna(constant_value)


def process_nan_values(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame = None,
    input_columns=None,
    output_columns=None,
    strategy: function = delete_strategy,
    constant_value=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handles missing values in the training dataset using
    the selected imputation or removal strategy.

    The function allows processing both input features (X_train)
    and target variables (y_train). The selected columns are
    combined temporarily to apply the chosen strategy.

    Available strategies:
        - mean: Replaces missing numerical values with the mean.
        - median: Replaces missing numerical values with the median.
        - delete_strategy: Removes rows containing missing values.
        - constant: Replaces missing values with a specified constant.

    Args:
        X_train (pd.DataFrame): Training dataset containing
            the input features.
        y_train (pd.Series or pd.DataFrame, optional): Training
            target variable(s). Defaults to None.
        input_columns (list, optional): Names of the input columns
            to be processed. Defaults to None.
        output_columns (list, optional): Names of the target columns
            to be processed. Defaults to None.
        strategy (function): Function defining the strategy used
            to handle missing values.
        constant_value (Any, optional): Constant value used when
            the constant strategy is selected. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - X_train_processed: Processed training input features.
            - y_train_processed: Processed training target variable(s),
              or None if y_train was not provided.

    Raises:
        ValueError: If the selected strategy is not valid.
        ValueError: If the constant strategy is selected without
            providing a constant value.
        ValueError: If no columns are selected.
        ValueError: If a selected column is not available
            in the training dataset.
    """
    valid_strategies = [mean, median, delete_strategy, constant]

    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy. Available options: {valid_strategies}")

    if strategy == constant and constant_value is None:
        raise ValueError("A constant value must be provided.")

    X_train = X_train.copy()

    if y_train is not None:
        y_train = y_train.copy()

    # Selected columns
    columns = list(dict.fromkeys((input_columns or []) + (output_columns or [])))

    if not columns:
        raise ValueError("At least one column must be selected.")

    # Check selected columns
    for column in columns:
        if column not in X_train.columns and (y_train is None or column not in getattr(y_train, "columns", [])):
            raise ValueError(f"Column '{column}' is not available in the training dataset.")

    # Combine X and y to process rows together
    if y_train is not None:
        train = pd.concat([X_train, y_train], axis=1)
    else:
        train = X_train.copy()

    # Apply the selected strategy
    if strategy is not constant:
        strategy(train, input_columns)
        strategy(y_train, input_columns)
    else:
        strategy(train, input_columns, constant_value)
        strategy(y_train, input_columns)

    # Separate X and y
    X_train_processed = train[X_train.columns]

    if y_train is not None:
        y_train_processed = train[y_train.name] if isinstance(y_train, pd.Series) else train[y_train.columns]

    else:
        y_train_processed = None

    return (X_train_processed, y_train_processed)


def detect_nan_values(X_train, y_train=None, input_columns=None, output_columns=None):
    """
    Detects and counts missing values in the selected columns
    of the training dataset.

    The function can inspect both input features (X_train)
    and target variables (y_train).

    It returns a summary indicating the number of missing values
    found in each selected column, as well as a boolean indicating
    whether any missing values exist.

    Args:
        X_train (pd.DataFrame): Training dataset containing
            the input features.
        y_train (pd.Series or pd.DataFrame, optional): Training
            target variable(s). Defaults to None.
        input_columns (list, optional): Names of the input columns
            to inspect. Defaults to None.
        output_columns (list, optional): Names of the target columns
            to inspect. Defaults to None.

    Returns:
        tuple[pd.DataFrame, bool]:
            - results: DataFrame containing the selected column names
              and their respective missing value counts.
            - has_missing_values: True if at least one missing value
              is found, False otherwise.

    Raises:
        ValueError: If no columns are selected.
        ValueError: If a selected column does not exist
            in the combined training dataset.
    """
    columns = []

    if input_columns:
        columns.extend(input_columns)

    if output_columns:
        columns.extend(output_columns)

    columns = list(dict.fromkeys(columns))

    if not columns:
        raise ValueError("At least one column must be selected.")

    train = X_train.copy()

    if y_train is not None:
        if isinstance(y_train, pd.DataFrame):
            train = pd.concat([train, y_train], axis=1)
        else:
            train[y_train.name or "target"] = y_train

    # Check that the selected columns exist
    for column in columns:
        if column not in train.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")

    results = []

    for column in columns:
        results.append(
            {
                "column": column,
                "nan_count": int(train[column].isna().sum()),
            }
        )

    results = pd.DataFrame(results)

    # Check whether missing values exist
    has_missing_values = results["nan_count"].sum() > 0

    return results, has_missing_values
