import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets based on the specified target column.

    Args:
        df : The input DataFrame containing features and target.
        target_column (str) : The name of the target column to be predicted.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: A tuple containing the training DataFrame and testing DataFrame.
    """
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Combine the features and target variables
    train_df = pd.concat([train_X, train_y.reset_index(drop=True)], axis=1)
    test_df = pd.concat([test_X, test_y.reset_index(drop=True)], axis=1)

    return train_df, test_df
