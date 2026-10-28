import pandas as pd

def data_shufler(frame:pd.DataFrame, seed:int, test_percent:int):

    frame = frame.copy()

    frame.sample(frac=1, random_state=seed).reset_index(drop=True)

    test_percent = round(len(frame)*test_percent, 0)

    df_test = frame.iloc[test_percent:].copy()
    df_train = frame.iloc[:test_percent].copy()

    return df_test, df_train
