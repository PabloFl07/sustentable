import pandas as pd


def count_nan(dataframe:pd.DataFrame)->tuple[pd.DataFrame, int]:
    '''
    count_nan() calculates nan values on a dataframe
    
    :param dataframe: pandas DataFrame
    :type dataframe: pd.DataFrame
    :return: returns a copy from the DataFrame with all values to False, except NaN values which are set True, and an int with de agregation of all NaN values
    :rtype: tuple[DataFrame, int]
    '''
    nan_values = dataframe.isna()
    return nan_values, nan_values.sum()

def mean(dataframe:pd.DataFrame):
    