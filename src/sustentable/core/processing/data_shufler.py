import pandas as pd


def data_shuffler(frame: pd.DataFrame, seed: int, test_percent: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shuffle a DataFrame and split it into train and test sets.

    Parameters
    ----------
    frame : pd.DataFrame
        Original DataFrame.
    seed : int
        Random seed for reproducibility.
    test_percent : float
        Fraction of rows assigned to the test set, between 0 and 1.

    Returns
    -------
    df_test : pd.DataFrame
        Test subset.
    df_train : pd.DataFrame
        Training subset.
    """
    if type(test_percent) is not float:
        raise TypeError("test_percent must be a float")
    elif type(frame) is not pd.DataFrame:
        raise TypeError("frame must be a Pandas Dataframe")
    elif not 0 <= test_percent <= 1:
        raise ValueError("test_percent must be between 0 and 1.")
    elif type(seed) is not int:
        raise TypeError("seed must be a int")

    frame = frame.sample(frac=1, random_state=seed)

    n_test = round(len(frame) * test_percent)

    df_test = frame.iloc[:n_test].reset_index(drop=True).copy()
    df_train = frame.iloc[n_test:].reset_index(drop=True).copy()

    return df_test, df_train
