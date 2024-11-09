import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def data_cleaning(df):
    """
    Cleans the input raw DataFrame by removing rows with missing values and duplicates.

    Args:
        df : The DataFrame to be cleaned.

    Returns:
        The cleaned DataFrame.
    """

    # Drop rows with missing values
    initial_shape = df.shape
    df.dropna(inplace=True)
    dropped_missing = initial_shape[0] - df.shape[0]
    print(f"Dropped rows with missing values: {dropped_missing}")

    # Remove duplicate values
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    dropped_duplicates = initial_shape[0] - df.shape[0]
    print(f"Dropped rows with duplicate values: {dropped_duplicates}")

    return df
